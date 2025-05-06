from multiprocessing import Value
import sys
from time import time
from typing import Any
import numpy as np
from stormpy import (
    Environment,  # type: ignore
    MinMaxMethod,  # type: ignore
    check_interval_mdp,  # type: ignore
    CheckTask,  # type: ignore
    ExpressionManager,  # type: ignore
    parse_properties,
    SparseIntervalModelComponents,  # type: ignore
    SparseIntervalMdp,  # type: ignore
    SparseIntervalPomdp,  # type: ignore
    StateLabeling,  # type: ignore
)
from stormpy.pomdp import (
    ObservationTraceUnfolderOptions,  # type: ignore
    ObservationTraceUnfolderInterval,  # type: ignore
)
from stormpy.pycarl import Interval  # type: ignore
import stormpy as sp

import argparse

from premise.monitor import UnfoldingRiskAssessment, Monitor


State = tuple[Any, Any, bool]
Trace = tuple[State, ...]
Samples = list[Trace]


def hamming_lookup(d: dict[str, int], key: str) -> int:
    # Find the value in the dict where the key has the closest hamming distance to the given key
    def hamming_distance(s1: str, s2: str) -> int:
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    return min(d.items(), key=lambda item: hamming_distance(item[0], key))[1]


def stormpy_pomdp_to_mdp(pomdp):
    components = SparseIntervalModelComponents(pomdp.transition_matrix, pomdp.labeling)
    if pomdp.has_choice_labeling():
        components.choice_labeling = pomdp.choice_labeling
    if pomdp.has_state_valuations():
        components.state_valuations = pomdp.state_valuations

    return SparseIntervalMdp(components)


def stormpy_product_unroll(i_mdp: SparseIntervalMdp, horizon):
    nr_of_horizon_levels = horizon + 1

    # Create the matrix
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
            current_row, i * nr_of_horizon_levels + horizon, Interval(1.0)
        )
        current_row += 1

    matrix = builder.build(
        overridden_column_count=len(i_mdp.states) * nr_of_horizon_levels
    )

    # Build the labeling
    labeling = StateLabeling(len(i_mdp.states) * nr_of_horizon_levels)
    for i in range(nr_of_horizon_levels):
        labeling.add_label(f"step={i}")
    for label in i_mdp.labeling.get_labels():
        labeling.add_label(label)
    labeling.add_label("horizon")

    for i in range(len(i_mdp.states)):
        for h in range(horizon + 1):
            for label in i_mdp.states[i].labels:
                if label == "init" and h == 0 or label != "init":
                    labeling.add_label_to_state(label, i * nr_of_horizon_levels + h)

            labeling.add_label_to_state("step=" + str(h), i * nr_of_horizon_levels + h)

        labeling.add_label_to_state("horizon", i * nr_of_horizon_levels + horizon)

    components = SparseIntervalModelComponents(matrix, labeling)
    return SparseIntervalMdp(components)


def dict_to_interval_ipomdp(
    trans_dict, init_dict, target_label
) -> tuple[Any, dict[Any, int], dict[Any, int]]:
    transitions: dict[int, dict[int, Interval]] = {}
    state_index_map: dict[Any, int] = {}
    observations = {}
    observation_map = {}

    # Get all states
    state_index = 1

    real_states = set()
    for (s, d), (l, u) in trans_dict.items():
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
        interval = Interval(l, u)
        transitions[s_index][d_index] = interval

    builder = sp.storage.IntervalSparseMatrixBuilder(0, 0, 0, False, True)
    current_row = 0
    for s, d_dict in sorted(transitions.items()):
        builder.new_row_group(current_row)
        if sum([x.upper() for x in d_dict.values()]) < 1:
            raise ValueError(
                f"Upper bounds are below 1 for {s} ({sum([x.upper() for x in d_dict.values()])}): {d_dict}"
            )
        if sum([x.lower() for x in d_dict.values()]) > 1:
            raise ValueError(
                f"Lower bounds are above 1 for {s} ({sum([x.lower() for x in d_dict.values()])}): {d_dict}"
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

    components = SparseIntervalModelComponents(matrix, labeling)
    components.observability_classes = [0] + [
        o for _, o in sorted(observations.items())
    ]
    return SparseIntervalPomdp(components), observation_map, state_index_map


class UnfoldingIntervalRiskAssessment(UnfoldingRiskAssessment):
    def __init__(self, stormpy_environment, unfolder, dump_model_to=None, maxmin="max"):
        self._stormpy_env = stormpy_environment
        self._unfolder = unfolder
        self._mdp = None
        self._dump_model_to = dump_model_to
        self._current_step = 0
        self._prop = sp.parse_properties(f'P{maxmin}=? [F "_goal"]')[0]

    def get_risk(self, deadline=None):
        sp.reset_timeout()  # type: ignore
        if deadline:
            sp.set_timeout(int(deadline / 1000))  # type: ignore
        try:
            task = CheckTask(self._prop.raw_formula, False)
            result = check_interval_mdp(self._mdp, task, self._stormpy_env)
            risk = result.at(self._mdp.initial_states[0])  # type: ignore
        except RuntimeError:
            print("What")
            return False, 0
        sp.reset_timeout()  # type: ignore
        return True, risk


def create_monitor(
    trans_dict, init_dict, maxmin, target_label, horizon, dump_path=None, verbose=0
) -> tuple[
    Monitor, dict[Any, int], ObservationTraceUnfolderInterval, SparseIntervalPomdp
]:
    stormpy_environment = Environment()
    stormpy_environment.solver_environment.minmax_solver_environment.method = (
        MinMaxMethod.value_iteration
    )

    # ipomdp = build_interval_model_from_drn("premise/examples/tiny-05.drn")

    ipomdp, observation_map, state_index_map = dict_to_interval_ipomdp(
        trans_dict, init_dict, target_label
    )
    if verbose > 1:
        print(ipomdp)
        with open("models/imc.dot", "w") as f:
            f.write(ipomdp.to_dot())

    options = ObservationTraceUnfolderOptions()
    options.rejection_sampling = True

    expr_manager = ExpressionManager()

    prop = parse_properties(f'P{maxmin}=? [F "target"]')

    task = CheckTask(prop[0].raw_formula, False)
    imdp = stormpy_pomdp_to_mdp(ipomdp)
    imdp = stormpy_product_unroll(imdp, horizon)
    result = check_interval_mdp(imdp, task, stormpy_environment)

    risks = []
    for i in range(len(ipomdp.states)):
        if maxmin == "min":
            risks.append(
                Interval(
                    max([result.at(i * (horizon + 1) + h) for h in range(horizon + 1)])
                )
            )
        else:
            risks.append(
                Interval(
                    min([result.at(i * (horizon + 1) + h) for h in range(horizon + 1)])
                )
            )

    if verbose > 0:
        for s, i in state_index_map.items():
            print(f"{s}= {risks[i]}")

    unfolder = ObservationTraceUnfolderInterval(
        ipomdp,
        risks,
        expr_manager,
        options,
    )

    ura = UnfoldingIntervalRiskAssessment(
        stormpy_environment, unfolder, dump_path, maxmin
    )

    mon = Monitor(ura, None)

    return mon, observation_map, unfolder, ipomdp


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

    args = parser.parse_args()

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
            elif action == "speed":
                t = time()
                for i in range(30):
                    print(mon.step(188), " -> ", end="")
                print(f"done in {time() - t}s")
            else:
                print(mon.step(int(hamming_lookup(observation_map, action))))
            print(
                "Observations with id:\n"
                + "\t".join([f"{k}: {v}" for k, v in observation_map.items()])
            )
            action = input("Next Step (\\d*/r/speed) ")
