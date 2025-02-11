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
    ChoiceLabeling,
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
        builder.add_next_value(current_row, i * nr_of_horizon_levels + horizon, Interval(1.0))
        current_row += 1

    matrix = builder.build(overridden_column_count=len(i_mdp.states) * nr_of_horizon_levels)

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


def stormpy_simulator_unroll(hmm: SparseIntervalMdp, horizon):
    states: dict[tuple[int, int], int] = {
        (0, s.id): new_s for new_s, s in enumerate(hmm.states)
    }
    state_labels: dict[int, set[str]] = {
        new_s: set(["step=0"]) for new_s in states.values()
    }
    for s in hmm.initial_states:
        state_labels[states[(0, s)]].add("init")
    labels = set(["step=0", "init", "horizon", "target"]).union(hmm.labeling.get_labels())
    for h in range(horizon + 1):
        labels.add(f"step={h}")

    # action_labels_map: dict[int, set[str]] = {}
    # action_labels = set(hmm.choice_labeling.get_labels())

    queue = [(i, new_s, s) for (i, new_s), s in states.items()]
    horizon_queue = []

    builder = sp.storage.IntervalSparseMatrixBuilder(0, 0, 0, False, True)

    current_row = 0
    while queue:
        i, new_s, s = queue.pop(0)
        builder.new_row_group(current_row)
        old_state = hmm.states[s]
        state_labels[new_s].update(old_state.labels.difference(["init"]))

        for action in old_state.actions:
            # action_labels_map[current_row] = action.labels

            new_row_dict: dict[int, Any] = {}
            for transition in action.transitions:
                dest_s = transition.column
                if (i + 1, dest_s) in states:
                    new_dest_s = states[(i + 1, dest_s)]
                else:
                    new_dest_s = len(states)
                    states[(i + 1, dest_s)] = new_dest_s
                    state_labels[new_dest_s] = set(["step=" + str(i + 1)])
                    if i + 1 >= horizon:
                        state_labels[new_dest_s].add("horizon")
                        horizon_queue.append((new_dest_s, dest_s))
                    else:
                        queue.append((i + 1, new_dest_s, dest_s))

                new_row_dict[new_dest_s] = transition.value()

            for new_dest_s, value in sorted(new_row_dict.items()):
                builder.add_next_value(current_row, new_dest_s, value)

            current_row += 1

    for new_s, s in horizon_queue:
        builder.new_row_group(current_row)
        old_state = hmm.states[s]
        state_labels[new_s].update(old_state.labels.difference(["init"]))

        for action in old_state.actions:
            # action_labels_map[current_row] = action.labels
            builder.add_next_value(current_row, new_s, Interval(1.0))
            current_row += 1

    matrix = builder.build(overridden_column_count=len(states))

    # Create state labeling
    labeling = StateLabeling(len(states))
    for label in labels:
        labeling.add_label(label)

    for state, labels in state_labels.items():
        for label in labels:
            labeling.add_label_to_state(label, state)

    # Create choice labeling
    # choice_labeling = ChoiceLabeling(len(action_labels_map))
    # for label in action_labels:
    #     choice_labeling.add_label(label)

    # for action, labels in action_labels_map.items():
    #     for label in labels:
    #         choice_labeling.add_label_to_choice(label, action)

    components = SparseIntervalModelComponents(matrix, labeling)
    # components.choice_labeling = choice_labeling
    return SparseIntervalMdp(components), states


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


def create_monitor(
    trans_dict, init_dict, maxmin, target_label, horizon, dump_path=None, verbose=1
):
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
    print(imdp)
    result = check_interval_mdp(imdp, task, stormpy_environment)

    risks = []
    for i in range(len(ipomdp.states)):
        if maxmin == "min":
            risks.append(Interval(max([result.at(i * (horizon + 1) + h) for h in range(horizon + 1)])))
        else:
            risks.append(Interval(min([result.at(i * (horizon + 1) + h) for h in range(horizon + 1)])))

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
        args.target,
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
                for i in range(100):
                    print(mon.step(2), " -> ", end="")
                print(f"done in {time() - t}s")
            else:
                print(mon.step(int(hamming_lookup(observation_map, action))))
            print(
                "Observations with id:\n"
                + "\t".join([f"{k}: {v}" for k, v in observation_map.items()])
            )
            action = input("Next Step (\\d*/r/speed) ")
