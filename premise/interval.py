"""
Monitoring with interval hidden Markov models (iHMMs), e.g. learned from samples.

The model is given as dictionaries mapping transitions to probability intervals. Every state is a tuple whose
second to last entry is the observation and whose last entry describes whether the state is a target (bad) state.
The model is turned into an interval POMDP with a single action per state, the risk of every state is the
(maximal or minimal) probability to reach a target state within the horizon, and traces are monitored by unfolding
them in the interval POMDP.

Ported from the interval branch, which required custom forks of Storm and stormpy; stormpy 2.0 supports
interval models natively. Note that Storm requires all intervals of interval models to have a positive lower bound.
"""

from dataclasses import dataclass
import logging
from typing import Any

import stormpy as sp
import stormpy.pomdp

from premise.monitor import Monitor, UnfoldingRiskAssessment

logger = logging.getLogger(__name__)

State = tuple[Any, ...]
Trace = tuple[State, ...]
Samples = list[Trace]
IntervalDict = dict[Any, tuple[float, float]]

INITIAL_STATE = 0
TARGET_LABEL = "target"


@dataclass
class MonitorComponents:
    monitor: Monitor
    # Maps the observations of the states to the observations of the interval POMDP.
    observation_map: dict[Any, int]
    unfolder: Any
    ipomdp: Any
    risks: list
    # Maps the states to the states of the interval POMDP.
    state_index_map: dict[State, int]
    # The observation of the artificial initial state, use it to initialize the monitor.
    initial_observation: int


def _value_types(use_exact):
    return (sp.RationalInterval, sp.Rational) if use_exact else (sp.Interval, float)


def _interval(lower, upper, use_exact):
    interval_type, number_type = _value_types(use_exact)
    if use_exact:
        return interval_type(sp.Rational(str(lower)), sp.Rational(str(upper)))
    return interval_type(float(lower), float(upper))


def _resolution_mode(maxmin: str):
    if maxmin == "max":
        return sp.UncertaintyResolutionMode.MAXIMIZE
    if maxmin == "min":
        return sp.UncertaintyResolutionMode.MINIMIZE
    raise ValueError(f"Unknown optimisation direction {maxmin}, use 'max' or 'min'.")


def dict_to_interval_ipomdp(
    trans_dict: dict[tuple[State, State], tuple[float, float]],
    init_dict: dict[State, tuple[float, float]],
    target_label,
    use_exact=True,
):
    """
    Builds an interval POMDP with a single action per state from interval dictionaries.

    The interval POMDP has an artificial initial state (with its own observation) whose successors are given by init_dict.

    :param trans_dict: Maps transitions (source, destination) to probability intervals (lower, upper).
    :param init_dict: Maps states to the interval of their initial probability.
    :param target_label: States whose last entry equals target_label are labelled as target.
    :param use_exact: Whether exact (rational) intervals are used.
    :return: The interval POMDP, the observation map and the state index map.
    """
    state_index_map: dict[State, int] = {}
    observation_map: dict[Any, int] = {}
    observations = [None]  # The observation of the initial state is set below.

    def index(state):
        if state not in state_index_map:
            if state[-2] not in observation_map:
                observation_map[state[-2]] = len(observation_map)
            state_index_map[state] = len(observations)
            observations.append(observation_map[state[-2]])
        return state_index_map[state]

    transitions: dict[int, dict[int, tuple[float, float]]] = {INITIAL_STATE: {}}
    for (source, dest), bounds in trans_dict.items():
        transitions.setdefault(index(source), {})[index(dest)] = bounds
        transitions.setdefault(index(dest), {})
    for state, bounds in init_dict.items():
        # States that are not part of any transition cannot be reached in the model.
        if state in state_index_map:
            transitions[INITIAL_STATE][state_index_map[state]] = bounds
    initial_observation = len(observation_map)
    observations[INITIAL_STATE] = initial_observation

    interval_type, _ = _value_types(use_exact)
    builder = sp.datastructures.SparseMatrixBuilder[interval_type](0, 0, 0, False, True)
    for state in range(len(observations)):
        successors = transitions[state]
        lower_sum = sum(lower for lower, _ in successors.values())
        upper_sum = sum(upper for _, upper in successors.values())
        if not successors or upper_sum < 1 or lower_sum > 1:
            raise ValueError(
                f"The intervals of state {state} do not contain a distribution (sum of lower bounds {lower_sum}, "
                f"sum of upper bounds {upper_sum}): {successors}"
            )
        builder.new_row_group(state)
        for dest, (lower, upper) in sorted(successors.items()):
            if lower <= 0:
                raise ValueError(f"Storm requires positive lower bounds, but transition {state} -> {dest} has [{lower}, {upper}].")
            builder.add_next_value(state, dest, _interval(lower, upper, use_exact))
    nr_states = len(observations)
    matrix = builder.build(nr_states, nr_states, nr_states)

    labeling = sp.models.StateLabeling(nr_states)
    labeling.add_label("init")
    labeling.add_label(TARGET_LABEL)
    labeling.add_label_to_state("init", INITIAL_STATE)
    for state, i in state_index_map.items():
        if state[-1] == target_label:
            labeling.add_label_to_state(TARGET_LABEL, i)

    components = sp.models.SparseModelComponents[interval_type](matrix, labeling)
    components.observability_classes = observations
    ipomdp = sp.models.SparsePomdp[interval_type](components)
    return ipomdp, observation_map, state_index_map, initial_observation


def compute_risks(ipomdp, maxmin, horizon, target=TARGET_LABEL, environment=None):
    """
    Computes the risk of every state: the maximal or minimal probability to reach a target state within the horizon.

    :return: The risks as (point) intervals, as required by the unfolder.
    """
    prop = sp.parse_property(f'P{maxmin}=? [F<={horizon} "{target}"]')
    result = sp.model_checking(
        ipomdp,
        prop,
        force_fully_observable=True,
        uncertainty_resolution_mode=_resolution_mode(maxmin),
        environment=environment,
    )
    interval_type = sp.RationalInterval if ipomdp.is_exact else sp.Interval
    return [interval_type(value) for value in result.get_value_vector()]


def build_monitor_from_model(ipomdp, maxmin, horizon, target=TARGET_LABEL, precision=None):
    """
    Builds a monitor for the interval POMDP.

    :param maxmin: Whether the uncertainty is resolved maximising ('max') or minimising ('min') the risk.
    :param horizon: The horizon of the risk.
    :param precision: (optional) Precision of the (robust) value iteration.
    :return: The monitor, the unfolder and the risk of every state.
    """
    environment = sp.Environment()
    if precision is not None:
        environment.solver().min_max().precision = sp.Rational(precision)

    risks = compute_risks(ipomdp, maxmin, horizon, target, environment)
    logger.info("Computed the risks for all states")

    unfolder = sp.pomdp.create_observation_trace_unfolder(
        ipomdp, risks, sp.expressions.ExpressionManager(), use_restart_semantics=True
    )
    risk_assessment = UnfoldingRiskAssessment(
        environment, unfolder, uncertainty_resolution_mode=_resolution_mode(maxmin)
    )
    return Monitor(risk_assessment, None), unfolder, risks


def create_monitor(
    trans_dict,
    init_dict,
    maxmin,
    target_label,
    horizon,
    use_exact=True,
    precision=None,
) -> tuple[Monitor, MonitorComponents]:
    """
    Creates a monitor for the interval HMM given by the interval dictionaries (see dict_to_interval_ipomdp).
    Initialize the monitor with the initial observation (MonitorComponents.initial_observation) and step it with
    the observations mapped by MonitorComponents.observation_map.
    """
    ipomdp, observation_map, state_index_map, initial_observation = dict_to_interval_ipomdp(
        trans_dict, init_dict, target_label, use_exact
    )
    logger.info(f"Created interval POMDP with {ipomdp.nr_states} states")
    mon, unfolder, risks = build_monitor_from_model(ipomdp, maxmin, horizon, precision=precision)
    return mon, MonitorComponents(
        mon, observation_map, unfolder, ipomdp, risks, state_index_map, initial_observation
    )
