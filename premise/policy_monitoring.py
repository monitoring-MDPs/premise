"""
Monitoring under a set of hypotheses about the policy of the monitored agent.

Each hypothesis is an objective (for now a property, later a reward function) for which the agent acts optimally.
The pipeline is (the steps are marked with "Step N" comments in the code):

1. Input: a list of properties (the objectives, one per hypothesis) and a POMDP model.
   See `benchmarks` and `build_pomdp`.
2. For each property, get the optimal policy (on the fully observable POMDP). See `optimal_policy`.
3. Apply this policy to get an HMM, i.e., a POMDP with a single action per state, which can be unfolded.
   See `induce_hmm`.
4. To monitor a trace, calculate in each HMM
   a. the risk (conditioned on the trace), and
   b. the probability of the trace.
   The risk of every state is computed once in `build_hypotheses`; the trace is unfolded in
   `PolicyMixtureMonitor.step`.
5. Normalise the trace probabilities (weighted by the prior) over the hypotheses to get a distribution
   over the reward functions. See `PolicyMixtureMonitor.step`.
6. Weigh the risk of each HMM with the probability of its hypothesis to get the weighted risk.
   See `PolicyMixtureMonitor.step`.
"""

import argparse
from dataclasses import dataclass, field
from fractions import Fraction
import logging
from pathlib import Path

import pandas as pd
import stormpy as sp
import stormpy.pomdp

from premise import models, trace_generator, traces
from premise.monitor import MonitorTimeOutException, UnfoldingRiskAssessment

logger = logging.getLogger(__name__)


@dataclass
class PolicyBenchmark:
    """
    A POMDP with a risk property and a set of hypotheses about the objective of the agent.
    """

    model_path: Path
    constants: str
    risk_property: str
    # Maps the name of a hypothesis to the objective for which the policy is optimal.
    hypotheses: dict[str, str]
    # (optional) The prior probability of every hypothesis, used in step 5. Values can be numbers or strings such as
    # "1/3"; they are normalised to sum to one. If None, the prior is uniform.
    priors: dict[str, float | int | str] | None = None


# Step 1: the inputs, a POMDP model and a list of properties (one per hypothesis).
benchmarks = {
    # The drone is the agent: it starts at the top in the middle and flies to target A (bottom left) or B (bottom right).
    # The drone reaches both targets with probability 1 from every state, so maximising the probability does not make
    # it fly to the target; instead, it minimises the expected number of moves to reach the target.
    "evade": PolicyBenchmark(
        models.EXAMPLES_DIR / "policy" / "evade-targets.nm",
        "N=5,RADIUS=3",
        'Pmax=? [F<=12 "crash"]',
        {
            "targetA": 'R{"dronemoves"}min=? [F "targetA"]',
            "targetB": 'R{"dronemoves"}min=? [F "targetB"]',
        },
    ),
    # The rover is the agent: it reaches the goal either via refuelling station A or via station B.
    "refuel": PolicyBenchmark(
        models.EXAMPLES_DIR / "policy" / "refuel-stations.nm",
        "N=6,ENERGY=8",
        'Pmax=? [F<=12 "empty"]',
        {
            "stationA": 'Pmax=? [F "goalA"]',
            "stationB": 'Pmax=? [F "goalB"]',
        },
    ),
}


@dataclass
class PolicyHypothesis:
    """
    The HMM obtained by applying the optimal policy for an objective, together with the risk of each of its states.
    """

    name: str
    hmm: sp.models.SparsePomdp
    risk: list
    prior: object


def load_program(benchmark: PolicyBenchmark, property_strings=()):
    """
    Parses the PRISM program of the benchmark and instantiates its constants (also in the given properties).

    :return: The instantiated program and the parsed properties.
    """
    program = sp.parse_prism_program(str(benchmark.model_path))
    props = [sp.parse_property(prop, program) for prop in property_strings]
    return sp.instantiate_constants(program, benchmark.constants, props)


def build_pomdp(benchmark: PolicyBenchmark, exact_arithmetic=True):
    """
    Step 1: builds the POMDP of the benchmark such that the risk property and all objectives can be checked on it.

    :return: The POMDP, the risk property and a dictionary mapping each hypothesis to its objective property.
    """
    names = list(benchmark.hypotheses)
    property_strings = [benchmark.risk_property] + [benchmark.hypotheses[name] for name in names]
    program, props = load_program(benchmark, property_strings)
    pomdp = models._build_model(program, props, exact_arithmetic=exact_arithmetic)
    assert pomdp.has_observation_valuations()
    return pomdp, props[0], dict(zip(names, props[1:]))


def optimal_policy(pomdp, objective):
    """
    Step 2: computes an optimal (memoryless, deterministic) policy for the objective,
    assuming the agent observes the full state.
    """
    result = sp.model_checking(pomdp, objective, force_fully_observable=True, extract_scheduler=True)
    scheduler = result.get_scheduler()
    if not scheduler.memoryless or not scheduler.deterministic:
        raise ValueError(f"The optimal policy for {objective} is not memoryless and deterministic.")
    return scheduler


def induce_hmm(pomdp, scheduler):
    """
    Step 3: applies a memoryless deterministic policy to the POMDP.

    :return: A POMDP with a single action per state (i.e., an HMM) with the observations and valuations of the POMDP.
    """
    value_type = sp.Rational if pomdp.is_exact else float
    matrix = pomdp.transition_matrix
    builder = sp.datastructures.SparseMatrixBuilder[value_type](0, 0, 0, False, True)
    has_choice_labeling = pomdp.has_choice_labeling()
    if has_choice_labeling:
        choice_labeling = sp.models.ChoiceLabeling(pomdp.nr_states)
        for label in pomdp.choice_labeling.get_labels():
            choice_labeling.add_label(label)

    for state in range(pomdp.nr_states):
        choice = scheduler.get_choice(state)
        # States in which the policy is irrelevant (e.g. where the objective is unreachable) may have no choice.
        action = choice.get_deterministic_choice() if choice.defined else 0
        row = matrix.get_row_group_start(state) + action
        builder.new_row_group(state)
        for entry in matrix.get_row(row):
            builder.add_next_value(state, entry.column, entry.value)
        if has_choice_labeling:
            for label in pomdp.choice_labeling.get_labels_of_choice(row):
                choice_labeling.add_label_to_choice(label, state)

    components = sp.models.SparseModelComponents[value_type](
        builder.build(pomdp.nr_states, pomdp.nr_states, pomdp.nr_states), pomdp.labeling
    )
    components.observability_classes = list(pomdp.observations)
    components.observation_valuations = pomdp.observation_valuations
    if pomdp.has_state_valuations():
        components.state_valuations = pomdp.state_valuations
    if has_choice_labeling:
        components.choice_labeling = choice_labeling
    return sp.models.SparsePomdp[value_type](components)


def prior_distribution(benchmark: PolicyBenchmark, exact_arithmetic=True) -> dict[str, object]:
    """
    The prior over the hypotheses of the benchmark (used in step 5), normalised to sum to one.

    :return: A dictionary mapping each hypothesis to its prior probability (a Rational if exact_arithmetic is set).
    """
    names = list(benchmark.hypotheses)
    if benchmark.priors is None:
        weights = {name: Fraction(1) for name in names}
    else:
        if set(benchmark.priors) != set(names):
            raise ValueError(f"The priors must be given for exactly the hypotheses {names}, got {list(benchmark.priors)}.")
        # Going via the string representation keeps e.g. 0.3 exactly 3/10.
        weights = {name: Fraction(str(benchmark.priors[name])) for name in names}
    if any(weight < 0 for weight in weights.values()) or sum(weights.values()) == 0:
        raise ValueError(f"The priors must be non-negative and not all zero, got {benchmark.priors}.")
    total = sum(weights.values())
    convert = sp.Rational if exact_arithmetic else float
    return {name: convert(weight / total) for name, weight in weights.items()}


def build_hypotheses(benchmark: PolicyBenchmark, exact_arithmetic=True) -> list[PolicyHypothesis]:
    """
    Builds an HMM for every hypothesis of the benchmark, computes the risk of its states and attaches its prior.
    """
    # The prior over the hypotheses, used in step 5 (validated before the expensive model building).
    priors = prior_distribution(benchmark, exact_arithmetic)
    # Step 1: the POMDP and the properties.
    logger.info(f"Building POMDP {benchmark.model_path}...")
    pomdp, risk_property, objectives = build_pomdp(benchmark, exact_arithmetic)
    logger.info(f"POMDP has {pomdp.nr_states} states and {pomdp.nr_observations} observations.")

    hypotheses = []
    for name, objective in objectives.items():
        logger.info(f"Computing the optimal policy for {name}: {objective}")
        # Step 2: the optimal policy for this property.
        policy = optimal_policy(pomdp, objective)
        # Step 3: apply the policy to get an HMM.
        hmm = induce_hmm(pomdp, policy)
        # Step 4a (preparation): the risk of every state of the HMM. While monitoring, the unfolding combines these
        # into the risk conditioned on the trace.
        # The HMM has a single action, hence the optimisation direction of the risk property does not matter.
        risk = sp.model_checking(hmm, risk_property, force_fully_observable=True).get_value_vector()
        hypotheses.append(PolicyHypothesis(name, hmm, risk, priors[name]))
    return hypotheses


@dataclass
class MixtureRiskEstimate:
    """
    The result of monitoring a trace under multiple hypotheses.
    """

    # The risk, weighted by the posterior of the hypotheses.
    risk: object
    # Per hypothesis: the posterior probability, the probability of the trace and the risk conditioned on the trace.
    posterior: dict[str, object] = field(default_factory=dict)
    trace_probability: dict[str, object] = field(default_factory=dict)
    conditional_risk: dict[str, object] = field(default_factory=dict)


class PolicyMixtureMonitor:
    """
    Monitors a trace under a set of hypotheses about the policy of the agent.
    """

    def __init__(self, hypotheses: list[PolicyHypothesis], stormpy_environment=None, deadline=None):
        self._hypotheses = hypotheses
        self._deadline = deadline
        environment = stormpy_environment if stormpy_environment is not None else sp.Environment()
        # Step 4 (preparation): one unfolder per HMM, used to compute the risk and the trace probability in that HMM.
        # Without restart semantics the unfolding keeps the probability of the trace (the probability to reach "_end").
        self._assessors = [
            UnfoldingRiskAssessment(
                environment,
                sp.pomdp.create_observation_trace_unfolder(
                    hypothesis.hmm,
                    hypothesis.risk,
                    sp.expressions.ExpressionManager(),
                    use_restart_semantics=False,
                ),
            )
            for hypothesis in hypotheses
        ]

    def initialize(self, observation):
        for assessor in self._assessors:
            assessor.initialize(observation)

    def step(self, observation) -> MixtureRiskEstimate:
        """
        Extends the trace by the observation and computes the weighted risk.

        The weighted risk is sum_i posterior_i * risk_i, where posterior_i is proportional to prior_i * P_i(trace).
        """
        estimate = MixtureRiskEstimate(risk=None)
        weights = {}
        # Step 4: for each HMM, extend the unfolding of the trace by the observation and compute
        for hypothesis, assessor in zip(self._hypotheses, self._assessors):
            assessor.step(observation)
            # Step 4b: the probability of the trace in this HMM.
            prob = self._value(assessor.get_prob(self._deadline))
            estimate.trace_probability[hypothesis.name] = prob
            weights[hypothesis.name] = hypothesis.prior * prob
            # Step 4a: the risk in this HMM, conditioned on the trace.
            # The conditional risk is undefined if the trace is impossible under this hypothesis.
            estimate.conditional_risk[hypothesis.name] = (
                self._value(assessor.get_risk(self._deadline)) if prob != 0 else None
            )

        # Step 5: normalise the (prior-weighted) trace probabilities over the hypotheses,
        # which gives the distribution over the reward functions.
        normalisation = sum(weights.values())
        if normalisation == 0:
            raise ValueError("The trace is impossible under all hypotheses.")
        estimate.risk = 0
        for name, weight in weights.items():
            estimate.posterior[name] = weight / normalisation
            # Step 6: weigh the risk of each HMM with the probability of its hypothesis.
            # Hypotheses under which the trace is impossible have probability 0 and contribute nothing.
            if weight != 0:
                estimate.risk = estimate.risk + estimate.posterior[name] * estimate.conditional_risk[name]
        return estimate

    @staticmethod
    def _value(status_and_value):
        status, value = status_and_value
        if not status:
            raise MonitorTimeOutException
        return value


def main():
    parser = argparse.ArgumentParser(description="Monitor a trace under multiple hypotheses about the policy of the agent.")
    parser.add_argument("--benchmark", choices=list(benchmarks), default="evade")
    parser.add_argument(
        "--true-hypothesis",
        help="The hypothesis whose HMM generates the trace (default: the first hypothesis of the benchmark).",
    )
    parser.add_argument("--trace-length", default=20, type=int, help="How long should the trace be?")
    parser.add_argument("--seed", default=0, type=int, help="Seed for the simulator.")
    parser.add_argument("--float", action="store_false", dest="exact", help="Use floating point numbers instead of exact arithmetic.")
    parser.add_argument("--verbose", action="store_true", help="Enable logging output.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    benchmark = benchmarks[args.benchmark]
    # Steps 1-3: build the POMDP with the properties (1), compute the optimal policy for every property (2)
    # and apply it to get an HMM (3). This also computes the risk of every HMM state, used in step 4a.
    hypotheses = build_hypotheses(benchmark, args.exact)
    names = [hypothesis.name for hypothesis in hypotheses]

    # The trace to monitor is simulated in the HMM of the "true" hypothesis (not part of the pipeline itself).
    true_name = args.true_hypothesis if args.true_hypothesis is not None else names[0]
    if true_name not in names:
        parser.error(f"Unknown hypothesis {true_name}, choose from {names}.")
    true_hypothesis = hypotheses[names.index(true_name)]
    generator = trace_generator.make_simulation_wrapper(true_hypothesis.hmm, args.trace_length, args.seed)
    mapper = traces.TraceMapper(true_hypothesis.hmm)

    # Step 4 (preparation): set up the unfolding of the trace in every HMM.
    monitor = PolicyMixtureMonitor(hypotheses)

    observation_names = mapper.get_observation_names()
    monitor.initialize(generator.initialize())
    rows = []
    while not generator.finished():
        observation = generator.step()
        # Steps 4-6 for the trace so far: the risk and trace probability in each HMM (4), the distribution over
        # the hypotheses (5) and the weighted risk (6).
        estimate = monitor.step(observation)

        # Write out the results, in the order of the steps, followed by the observation.
        row = {f"risk({name})": _to_float(estimate.conditional_risk[name]) for name in names}
        row |= {f"P({name})": _to_float(estimate.posterior[name]) for name in names}
        row["weighted risk"] = _to_float(estimate.risk)
        row |= dict(zip(observation_names, mapper.trace_to_high_level([observation])[0]))
        rows.append(row)

    results = pd.DataFrame(rows, index=pd.RangeIndex(1, len(rows) + 1, name="step"))
    print(f"Trace generated by hypothesis '{true_name}'.")
    print(results.to_string(float_format="{:.4f}".format, na_rep="-"))


def _to_float(value):
    # Undefined values (e.g. the risk under a hypothesis for which the trace is impossible) become NaN.
    return float("nan") if value is None else float(value)


if __name__ == "__main__":
    main()
