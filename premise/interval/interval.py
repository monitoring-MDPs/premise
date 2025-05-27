from time import time
from typing import Any
import numpy as np
import argparse

from pypoman import compute_polytope_vertices
from stormpy import (
    Environment,
    MinMaxMethod,
    check_interval_mdp,
    CheckTask,
    ExpressionManager,
    parse_properties,
    SparseIntervalModelComponents,
    SparseIntervalMdp,
    SparseIntervalPomdp,
    StateLabeling,
    SparseExactMdp,
    SparseExactModelComponents,
    SparseRationalIntervalModelComponents,
    SparseRationalIntervalMdp,
    SparseRationalIntervalPomdp,
    ExactCheckTask,
    check_exact_interval_mdp,
    SparseModelComponents,
    SparseMdp,
    SparseRationalIntervalDtmc,
    SparseIntervalDtmc,
)
from stormpy.pomdp import (
    ObservationTraceUnfolderOptions,
    ObservationTraceUnfolderInterval,
    ObservationTraceUnfolderRationalInterval,
)
from stormpy.pycarl import Interval
from stormpy.pycarl.gmp import Interval as RationalInterval, Rational
import stormpy as sp

from premise.interval.policy_iteration import policy_iter_imc
from premise.monitor import UnfoldingRiskAssessment, Monitor


State = tuple[Any, Any, bool]
Trace = tuple[State, ...]
Samples = list[Trace]


def hamming_lookup(d: dict[str, int], key: str) -> int:
    # Find the value in the dict where the key has the closest hamming distance to the given key
    def hamming_distance(s1: str, s2: str) -> int:
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    return min(d.items(), key=lambda item: hamming_distance(item[0], key))[1]


def stormpy_ipomdp_to_imdp(pomdp):
    if pomdp.is_exact:
        components = SparseRationalIntervalModelComponents(
            pomdp.transition_matrix, pomdp.labeling
        )
        if pomdp.has_choice_labeling():
            components.choice_labeling = pomdp.choice_labeling
        if pomdp.has_state_valuations():
            components.state_valuations = pomdp.state_valuations

        return SparseRationalIntervalMdp(components)
    else:
        components = SparseIntervalModelComponents(
            pomdp.transition_matrix, pomdp.labeling
        )
        if pomdp.has_choice_labeling():
            components.choice_labeling = pomdp.choice_labeling
        if pomdp.has_state_valuations():
            components.state_valuations = pomdp.state_valuations

        return SparseIntervalMdp(components)


def stormpy_imdp_to_imc(imdp):
    if imdp.is_exact:
        components = SparseRationalIntervalModelComponents(
            imdp.transition_matrix, imdp.labeling
        )
        components.transition_matrix.make_row_grouping_trivial()

        if imdp.has_choice_labeling():
            components.choice_labeling = imdp.choice_labeling
        if imdp.has_state_valuations():
            components.state_valuations = imdp.state_valuations

        return SparseRationalIntervalDtmc(components)
    else:
        components = SparseIntervalModelComponents(
            imdp.transition_matrix, imdp.labeling
        )
        components.transition_matrix.make_row_grouping_trivial()

        if imdp.has_choice_labeling():
            components.choice_labeling = imdp.choice_labeling
        if imdp.has_state_valuations():
            components.state_valuations = imdp.state_valuations

        return SparseIntervalDtmc(components)


def stormpy_exact_pomdp_to_mdp(pomdp):
    components = SparseExactModelComponents(pomdp.transition_matrix, pomdp.labeling)
    if pomdp.has_choice_labeling():
        components.choice_labeling = pomdp.choice_labeling
    if pomdp.has_state_valuations():
        components.state_valuations = pomdp.state_valuations

    return SparseExactMdp(components)


def stormpy_imdp_to_ipomdp(mdp, observations, observation_valuations=None):
    if mdp.is_exact:
        components = SparseRationalIntervalModelComponents(
            mdp.transition_matrix, mdp.labeling
        )
        components.observability_classes = observations
        if observation_valuations:
            components.observation_valuations = observation_valuations
        if mdp.has_choice_labeling():
            components.choice_labeling = mdp.choice_labeling
        if mdp.has_state_valuations():
            components.state_valuations = mdp.state_valuations

        return SparseRationalIntervalPomdp(components)
    else:
        components = SparseIntervalModelComponents(mdp.transition_matrix, mdp.labeling)
        components.observability_classes = observations
        if observation_valuations:
            components.observation_valuations = observation_valuations
        if mdp.has_choice_labeling():
            components.choice_labeling = mdp.choice_labeling
        if mdp.has_state_valuations():
            components.state_valuations = mdp.state_valuations

        return SparseIntervalPomdp(components)


def enumerate_imc_in_mdp(imc):
    """
    This does not work just yet!
    """
    if imc.is_exact:
        builder = sp.storage.ExactSparseMatrixBuilder(0, 0, 0, False, True)
    else:
        builder = sp.storage.SparseMatrixBuilder(0, 0, 0, False, True)

    current_row = 0
    for i in range(len(imc.states)):
        old_state = imc.states[i]

        # Get transitions for state
        transitions = []
        for action in old_state.actions:
            for transition in action.transitions:
                transitions.append(
                    (
                        transition.column,
                        transition.value().lower(),
                        transition.value().upper(),
                    )
                )

        if len(transitions) == 1:
            points = [[1.0]]
        else:
            num_vars = len(transitions)
            # Build the Linear system of inequalities to get vertices
            A = np.zeros((num_vars * 2 + 1, num_vars))
            b = np.zeros((num_vars * 2 + 1,))
            for i, (_, l, u) in enumerate(transitions):
                A[i * 2, i] = -1
                b[i * 2] = float(l)
                A[i * 2 + 1, i] = 1
                b[i * 2 + 1] = -float(u)
            A[-1, :].fill(1)
            b[-1] = -1.0

            # # Calculate interior point
            # norm_vector = np.reshape(
            #     np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1)
            # )
            # c = np.zeros((halfspaces.shape[1],))
            # c[-1] = -1
            # A = np.hstack((halfspaces[:, :-1], norm_vector))
            # b = -halfspaces[:, -1:]
            # interrior_point = linprog(c, A_ub=A, b_ub=b, bounds=(None, None)).x[:-1]
            #
            # # Calculate vertices
            # hs = HalfspaceIntersection(halfspaces, interrior_point)
            #
            # # Build new actions
            # points = hs.intersections

            points = compute_polytope_vertices(A, b)
            print(transitions, points)

        builder.new_row_group(current_row)
        for probs in points:
            if abs(sum(probs) - 1) > 0.0000001:
                continue
            new_trans = {}

            for i, p in enumerate(probs):
                new_trans[transitions[i][0]] = Rational(p) if imc.is_exact else p

            for dest, p in sorted(new_trans.items()):
                builder.add_next_value(current_row, dest, p)

            current_row += 1

    matrix = builder.build(overridden_column_count=len(imc.states))

    # Build the labeling
    labeling = StateLabeling(len(imc.states))
    for label in imc.labeling.get_labels():
        labeling.add_label(label)
    labeling.add_label("horizon")

    for i in range(len(imc.states)):
        for label in imc.states[i].labels:
            labeling.add_label_to_state(label, i)

    if imc.is_exact:
        components = SparseExactModelComponents(matrix, labeling)
        return SparseExactMdp(components)
    else:
        components = SparseModelComponents(matrix, labeling)
        return SparseMdp(components)


def stormpy_product_unroll(i_mdp: SparseIntervalMdp, horizon):
    nr_of_horizon_levels = horizon + 1

    # Create the matrix
    if i_mdp.is_exact:
        builder = sp.storage.RationalIntervalSparseMatrixBuilder(0, 0, 0, False, True)
    else:
        builder = sp.storage.IntervalSparseMatrixBuilder(0, 0, 0, False, True)

    current_row = 0
    for i in range(len(i_mdp.states)):
        old_state = i_mdp.states[i]
        for h in range(horizon):  # TODO: Check if horizon + 1 is correct
            builder.new_row_group(current_row)
            for action in old_state.actions:
                for transition in action.transitions:
                    dest_state = transition.column * nr_of_horizon_levels + h + 1
                    builder.add_next_value(current_row, dest_state, transition.value())

                current_row += 1

        # Horizon states just containt self loops
        builder.new_row_group(current_row)
        builder.add_next_value(
            current_row,
            i * nr_of_horizon_levels + horizon,
            RationalInterval(Rational(1.0)) if i_mdp.is_exact else Interval(1.0),
        )
        current_row += 1

    matrix = builder.build(
        overridden_column_count=len(i_mdp.states) * nr_of_horizon_levels
    )

    # Build the labeling
    labeling = StateLabeling(len(i_mdp.states) * nr_of_horizon_levels)
    for label in i_mdp.labeling.get_labels():
        labeling.add_label(label)
    labeling.add_label("horizon")

    for i in range(len(i_mdp.states)):
        for h in range(horizon + 1):
            for label in i_mdp.states[i].labels:
                if label == "init" and h == 0 or label != "init":
                    labeling.add_label_to_state(label, i * nr_of_horizon_levels + h)

        labeling.add_label_to_state("horizon", i * nr_of_horizon_levels + horizon)

    if i_mdp.is_exact:
        components = SparseRationalIntervalModelComponents(matrix, labeling)
        return SparseRationalIntervalMdp(components)
    else:
        components = SparseIntervalModelComponents(matrix, labeling)
        return SparseIntervalMdp(components)


def dict_to_interval_ipomdp(
    trans_dict: dict[tuple[State, State], tuple[float, float]],
    init_dict,
    target_label,
    use_exact=True,
) -> tuple[
    SparseIntervalPomdp | SparseRationalIntervalPomdp,
    dict[State, int],
    dict[State, int],
]:
    transitions: dict[int, dict[int, Interval]] = {}
    state_index_map: dict[State, int] = {}
    observations = {}
    observation_map = {}

    # Get all states
    state_index = 1

    real_states = set()
    for (s, d), _ in trans_dict.items():
        real_states.add(s)
        real_states.add(d)

        if d not in state_index_map:
            # Add not seen observations to the observation map
            if d[-2] not in observation_map:
                observation_map[d[-2]] = len(observation_map)

            state_index_map[d] = state_index
            transitions[state_index] = {}
            observations[state_index] = observation_map[d[-2]]
            state_index += 1

        if s not in state_index_map:
            # Add not seen observations to the observation map
            if s[-2] not in observation_map:
                observation_map[s[-2]] = len(observation_map)

            state_index_map[s] = state_index
            transitions[state_index] = {}
            observations[state_index] = observation_map[s[-2]]
            state_index += 1

    init_state = 0
    transitions[init_state] = {}
    for d, (l, u) in sorted(
        init_dict.items(),
        key=lambda x: (x[0][0][0] if isinstance(x[0][0], tuple) else x[0][0]),
    ):
        if d not in real_states:
            continue

        if use_exact:
            interval = RationalInterval(Rational(l), Rational(u))
        else:
            interval = Interval(l, u)
        transitions[init_state][state_index_map[d]] = interval

    for (s, d), (l, u) in sorted(
        trans_dict.items(),
        key=lambda x: x[0][0][0][0] if isinstance(x[0][0][0], tuple) else x[0][0][0],
    ):
        if s not in state_index_map:
            state_index_map[s] = state_index
            transitions[state_index] = {}
            observations[state_index] = observation_map[s[-2]]
            state_index += 1
        if d not in state_index_map:
            state_index_map[d] = state_index
            transitions[state_index] = {}
            observations[state_index] = observation_map[d[-2]]
            state_index += 1

        s_index = state_index_map[s]
        d_index = state_index_map[d]
        if use_exact:
            interval = RationalInterval(Rational(l), Rational(u))
        else:
            interval = Interval(l, u)
        transitions[s_index][d_index] = interval

    if use_exact:
        builder = sp.storage.RationalIntervalSparseMatrixBuilder(0, 0, 0, False, True)
    else:
        builder = sp.storage.IntervalSparseMatrixBuilder(0, 0, 0, False, True)

    current_row = 0
    for s, d_dict in sorted(transitions.items()):
        builder.new_row_group(current_row)
        if sum([x.upper() for x in d_dict.values()]) < 1:
            raise ValueError(
                f"Upper bounds are below 1 for state {s} ({sum([x.upper() for x in d_dict.values()])}): {d_dict}"
            )
        if sum([x.lower() for x in d_dict.values()]) > 1:
            raise ValueError(
                f"Lower bounds are above 1 for state {s} ({sum([x.lower() for x in d_dict.values()])}): {d_dict}"
            )
        for dest, interval in sorted(d_dict.items()):
            builder.add_next_value(current_row, dest, interval)
        current_row += 1

    matrix = builder.build(overridden_column_count=len(state_index_map) + 1)

    labeling = StateLabeling(len(state_index_map) + 1)  # For the initial state

    labeling.add_label("init")
    labeling.add_label("target")
    # for s, i in state_index_map.items():
    #     labeling.add_label(str(s))

    labeling.add_label_to_state("init", init_state)
    for s, i in state_index_map.items():
        # labeling.add_label_to_state(str(s), i)
        if s[-1] == target_label:  # target label (Change between models)
            labeling.add_label_to_state("target", i)

    if use_exact:
        components = SparseRationalIntervalModelComponents(matrix, labeling)
    else:
        components = SparseIntervalModelComponents(matrix, labeling)
    components.observability_classes = [0] + [
        o for _, o in sorted(observations.items())
    ]

    if use_exact:
        return SparseRationalIntervalPomdp(components), observation_map, state_index_map
    else:
        return SparseIntervalPomdp(components), observation_map, state_index_map


class UnfoldingIntervalRiskAssessment(UnfoldingRiskAssessment):
    def __init__(
        self,
        stormpy_environment,
        unfolder,
        dump_model_to=None,
        maxmin="min",
        method="PI",
    ):
        self._stormpy_env = stormpy_environment
        self._unfolder = unfolder
        self._mdp = None
        self._dump_model_to = dump_model_to
        self._current_step = 0
        self._prop = sp.parse_properties(f'P{maxmin}=? [F "_goal"]')[0]
        self._method = method

    def get_risk(self, deadline=None):
        sp.reset_timeout()
        if deadline:
            sp.set_timeout(int(deadline / 1000))
        try:
            if self._method == "PI":
                imc = stormpy_imdp_to_imc(self._mdp)
                result = policy_iter_imc(imc, self._prop, storm_env=self._stormpy_env)
                risk = float(result.at(imc.initial_states[0]))
            else:
                if self._mdp.is_exact:
                    task = ExactCheckTask(self._prop.raw_formula, False)
                    result = check_exact_interval_mdp(
                        self._mdp, task, self._stormpy_env
                    )
                    risk = float(result.at(self._mdp.initial_states[0]))
                else:
                    task = CheckTask(self._prop.raw_formula, False)
                    result = check_interval_mdp(self._mdp, task, self._stormpy_env)
                    risk = result.at(self._mdp.initial_states[0])
        except RuntimeError:
            print("Timeout occurred in risk assesment")
            return False, 0
        sp.reset_timeout()
        return True, risk


def create_monitor(
    trans_dict,
    init_dict,
    maxmin,
    target_label,
    horizon,
    dump_path=None,
    verbose=0,
    use_exact=True,
    precision=1e-6,
) -> tuple[
    Monitor, dict[Any, int], ObservationTraceUnfolderInterval, SparseIntervalPomdp
]:
    ipomdp, observation_map, state_index_map = dict_to_interval_ipomdp(
        trans_dict, init_dict, target_label, use_exact
    )
    if verbose > 1:
        print(ipomdp)
        with open("out/imc.dot", "w") as f:
            f.write(ipomdp.to_dot())

    mon, unfolder = build_monitor_from_model(
        ipomdp,
        maxmin,
        horizon,
        state_index_map,
        dump_path,
        verbose=verbose,
        precision=precision,
    )

    return mon, observation_map, unfolder, ipomdp


def build_monitor_from_model(
    ipomdp,
    maxmin,
    horizon,
    state_index_map=None,
    dump_path=None,
    target="target",
    verbose=0,
    precision=1e-6,
    method="PI",
):
    stormpy_environment = Environment()
    stormpy_environment.solver_environment.minmax_solver_environment.method = (
        MinMaxMethod.value_iteration
    )
    stormpy_environment.solver_environment.minmax_solver_environment.precision = (
        Rational(precision)
    )

    options = ObservationTraceUnfolderOptions()
    options.rejection_sampling = True

    expr_manager = ExpressionManager()

    prop = parse_properties(f'P{maxmin}=? [F "{target}"]')

    if ipomdp.is_exact:
        task = ExactCheckTask(prop[0].raw_formula, False)
    else:
        task = CheckTask(prop[0].raw_formula, False)

    imdp = stormpy_ipomdp_to_imdp(ipomdp)
    imdp = stormpy_product_unroll(imdp, horizon)

    if ipomdp.is_exact:
        result = check_exact_interval_mdp(imdp, task, stormpy_environment)
    else:
        result = check_interval_mdp(imdp, task, stormpy_environment)

    # Risks are the risks at the step 0 of every state, they are experimentally checked correct
    risks = []
    for i in range(len(ipomdp.states)):
        if ipomdp.is_exact:
            risks.append(RationalInterval(result.at(i * (horizon + 1))))
        else:
            risks.append(Interval(result.at(i * (horizon + 1))))

    if verbose > 0:
        if state_index_map is None:
            for s, r in enumerate(risks):
                print(f"{s}= {float(r.upper())}")
        else:
            for s, i in state_index_map.items():
                print(f"{s}= {float(risks[i].upper())}")

    # If method is PI we don't want to set the minmaxmethod, thus we remake it:
    if method == "PI":
        stormpy_environment = Environment()
        stormpy_environment.solver_environment.minmax_solver_environment.precision = (
            Rational(precision)
        )

    if ipomdp.is_exact:
        unfolder = ObservationTraceUnfolderRationalInterval(
            ipomdp,
            risks,
            expr_manager,
            options,
        )
    else:
        unfolder = ObservationTraceUnfolderInterval(
            ipomdp,
            risks,
            expr_manager,
            options,
        )

    ura = UnfoldingIntervalRiskAssessment(
        stormpy_environment, unfolder, dump_path, maxmin, method
    )

    mon = Monitor(ura, None)
    return mon, unfolder


def main(args):
    trans_dict = np.load(args.trans_path, allow_pickle=True)[()]
    init_dict = np.load(args.init_path, allow_pickle=True)[()]
    mon, observation_map, unfolder, ipomdp = create_monitor(
        trans_dict,
        init_dict,
        args.maxmin,
        bool(args.target),
        args.horizon,
        args.dump,
        args.verbose,
    )
    import os

    if args.verbose > 0:
        print(os.getpid())

    if args.trace:
        traces = np.load(args.trace, allow_pickle=True)[()]
        for trace in traces.values():
            mon.initialize(0)
            observations = [t[-2] for t in trace]
            if args.verbose > 0:
                print()
            if args.verbose > 0:
                print(observations)
            last_risk = None
            for obs in observations:
                last_risk = mon.step(observation_map[obs])
                if args.verbose > 0:
                    print(last_risk, end=" -> ")

            if args.verbose == 0:
                print(last_risk)
    else:
        action = "r"
        while True:
            if action == "r":
                mon.initialize(0)
            elif action.isdigit():
                print(mon.step(int(action)))
            elif action[:5] == "speed":
                _, obs, rep = action.split(" ")
                t = time()
                for i in range(int(rep)):
                    print(mon.step(int(obs)), " -> ", end="")
                print(f"done in {time() - t}s")
            else:
                print(mon.step(int(hamming_lookup(observation_map, action))))
            print(
                "Observations with id:\n"
                + "\t".join([f"{k}: {v}" for k, v in observation_map.items()])
            )
            action = input("Next Step (\\d*/r/speed) ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interval POMDP Monitor")
    parser.add_argument(
        "trans_path", type=str, help="Path to the transition dictionary"
    )
    parser.add_argument(
        "init_path", type=str, help="Path to the initial state dictionary"
    )

    parser.add_argument(
        "--maxmin",
        type=str,
        default="min",
        choices=["max", "min"],
        help="Max or Min method for risk assessment (Min is maximizing the interval (while choosing the min action), Max is minimizing (while choosing the max action))",
    )
    parser.add_argument(
        "--trace", type=str, help="Path to the trace file for static mode"
    )
    parser.add_argument(
        "--dump", type=str, help="Path to the file to dump the model to"
    )
    parser.add_argument("--target", type=str, help="The target label to check for")
    parser.add_argument("--horizon", type=int, help="The horizon to monitor on")
    parser.add_argument("--verbose", "-v", action="count", default=0)

    int_args = parser.parse_args()

    main(int_args)
