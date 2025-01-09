from cProfile import label
from ipaddress import ip_address
from mimetypes import init
import sys
from time import time
from typing import Any
import numpy as np
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
)
from stormpy.pomdp import (
    ObservationTraceUnfolderOptions,
    ObservationTraceUnfolderInterval,
)
from stormpy.pycarl import Interval
import stormpy as sp

import monitor
import argparse


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


def dict_to_interval_ipomdp(trans_dict, init_dict, target_label):
    transitions: dict[int, dict[int, Interval]] = {}
    state_index_map: dict[Any, int] = {}
    observations = {}
    observation_map = {}

    real_states = set()
    for (s, d), (l, u) in trans_dict.items():
        real_states.add(s)
        real_states.add(d)

    init_state = 0
    state_index = 1
    transitions[init_state] = {}

    for d, (l, u) in init_dict.items():
        if d not in real_states:
            continue

        if d not in state_index_map:
            # Add not seen observations to the observation map
            if d[-2] not in observation_map:
                observation_map[d[-2]] = len(observation_map)

            state_index_map[d] = state_index
            transitions[state_index] = {}
            observations[state_index] = observation_map[d[-2]]
            state_index += 1

        interval = Interval(l, u)
        transitions[init_state][state_index_map[d]] = interval

    for (s, d), (l, u) in trans_dict.items():
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
        for trans_dict, interval in sorted(d_dict.items()):
            builder.add_next_value(current_row, trans_dict, interval)
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
        if s[-1] == target_label: #target label (Change between models)
            labeling.add_label_to_state("target", i)

    components = SparseIntervalModelComponents(matrix, labeling)
    components.observability_classes = [0] + [
        o for _, o in sorted(observations.items())
    ]
    return SparseIntervalPomdp(components), observation_map


class UnfoldingIntervalRiskAssessment(monitor.UnfoldingRiskAssessment):
    def __init__(self, stormpy_environment, unfolder, dump_model_to=None, maxmin="max"):
        self._stormpy_env = stormpy_environment
        self._unfolder = unfolder
        self._mdp = None
        self._dump_model_to = dump_model_to
        self._current_step = 0
        self._prop = sp.parse_properties(f'P{maxmin}=? [F "_goal"]')[0]

    def get_risk(self, deadline=None):
        sp.reset_timeout()
        if deadline:
            sp.set_timeout(int(deadline / 1000))
        try:
            task = CheckTask(self._prop.raw_formula, False)
            result = check_interval_mdp(self._mdp, task, self._stormpy_env)
            risk = result.at(self._mdp.initial_states[0])
        except RuntimeError:
            print("What")
            return False, 0
        sp.reset_timeout()
        return True, risk


def create_monitor(trans_dict, init_dict, maxmin, target_label, dump_path=None, verbose=1):
    stormpy_environment = Environment()
    stormpy_environment.solver_environment.minmax_solver_environment.method = (
        MinMaxMethod.value_iteration
    )

    # ipomdp = build_interval_model_from_drn("premise/examples/tiny-05.drn")

    ipomdp, observation_map = dict_to_interval_ipomdp(trans_dict, init_dict, target_label)
    if verbose > 1:
        print(ipomdp)
        with open("models/imc.dot", "w") as f:
            f.write(ipomdp.to_dot())

    options = ObservationTraceUnfolderOptions()
    options.rejection_sampling = True

    expr_manager = ExpressionManager()

    prop = parse_properties(f'P{maxmin}=? ["target"]')

    task = CheckTask(prop[0].raw_formula, False)
    imdp = stormpy_pomdp_to_mdp(ipomdp)
    risk_assessment = check_interval_mdp(imdp, task, stormpy_environment)
    risk_assessment = [
        Interval(risk_assessment.at(i)) for i in range(len(ipomdp.states))
    ]
    # print("risk=", risk_assessment, type(risk_assessment[0]))
    unfolder = ObservationTraceUnfolderInterval(
        ipomdp,
        risk_assessment,
        expr_manager,
        options,
    )

    ura = UnfoldingIntervalRiskAssessment(
        stormpy_environment, unfolder, dump_path, maxmin
    )

    mon = monitor.Monitor(ura, None)

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
        help="Max or Min method for risk assessment",
    )
    parser.add_argument(
        "--trace", type=str, help="Path to the trace file for static mode"
    )
    parser.add_argument(
        "--dump", type=str, help="Path to the file to dump the model to"
    )
    parser.add_argument("--target", type=str, help="The target label to check for")
    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()

    trans_dict = np.load(args.trans_path, allow_pickle=True)[()]
    init_dict = np.load(args.init_path, allow_pickle=True)[()]
    mon, observation_map, unfolder, ipomdp = create_monitor(
        trans_dict, init_dict, args.maxmin, args.target, args.dump, args.verbose
    )
    import os

    print(os.getpid())

    if args.trace:
        traces = np.load(args.trace, allow_pickle=True)[()]
        for trace in traces.values():
            mon.initialize(0)
            observations = [t[-2] for t in trace]
            print()
            if args.verbose > 0:
                print(observations)
            for obs in observations:
                print(mon.step(observation_map[obs]), end=" -> ")
    else:
        action = "r"
        while True:
            if action == "r":
                mon.initialize(0)
            elif action.isdigit():
                print(mon.step(int(action)))
            elif action == "speed":
                t = time()
                for i in range(100):
                    print(mon.step(2), " -> ", end="")
                print(f"done in {time() - t}s")
            else:
                print(mon.step(int(hamming_lookup(observation_map, action))))
            print(observation_map, ipomdp)
            action = input("Next Step (\\d*/r/speed) ")
