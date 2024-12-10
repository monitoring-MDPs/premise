from cProfile import label
import sys
from time import time
import numpy as np
from stormpy import (
    build_interval_model_from_drn,
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
import stormpy as sp
from pycarl import Interval

import monitor
import os

print(os.getpid())


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


def dict_to_interval_ipomdp(trans_dict, init_dict):
    transitions: dict[int, dict[int, Interval]] = {}
    state_index_map: dict[str, int] = {}
    observation_map = {"far": 1, "close": 2}
    observations = {}

    init_state = 0
    state_index = 1
    transitions[init_state] = {}

    for d, (l, u) in init_dict.items():
        if d not in state_index_map:
            state_index_map[d] = state_index
            transitions[state_index] = {}
            observations[state_index] = observation_map[d.split("_")[2]]
            state_index += 1

        interval = Interval(l, u)
        transitions[init_state][state_index_map[d]] = interval

    for (s, d), (l, u) in trans_dict.items():
        if s not in state_index_map:
            state_index_map[s] = state_index
            transitions[state_index] = {}
            observations[state_index] = observation_map[s.split("_")[2]]
            state_index += 1
        if d not in state_index_map:
            state_index_map[d] = state_index
            transitions[state_index] = {}
            observations[state_index] = observation_map[d.split("_")[2]]
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
    for s, i in state_index_map.items():
        labeling.add_label(s)

    labeling.add_label_to_state("init", 0)
    labeling.add_label_to_state("target", state_index_map["00_00_close"])
    for s, i in state_index_map.items():
        labeling.add_label_to_state(s, i)

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


print(os.getpid())


stormpy_environment = Environment()
stormpy_environment.solver_environment.minmax_solver_environment.method = (
    MinMaxMethod.value_iteration
)

# ipomdp = build_interval_model_from_drn("premise/examples/tiny-05.drn")
trans_dict = np.load(sys.argv[1], allow_pickle=True)[()]
init_dict = np.load(sys.argv[2], allow_pickle=True)[()]
ipomdp, observation_map = dict_to_interval_ipomdp(trans_dict, init_dict)
print(ipomdp)
with open("models/imc.dot", "w") as f:
    f.write(ipomdp.to_dot())

options = ObservationTraceUnfolderOptions()
options.rejection_sampling = True

expr_manager = ExpressionManager()

mm = input("max or min (max means choose best action but worst interval)? ")
prop = parse_properties(f'P{mm}=? ["target"]')

task = CheckTask(prop[0].raw_formula, False)
imdp = stormpy_pomdp_to_mdp(ipomdp)
risk_assessment = check_interval_mdp(imdp, task, stormpy_environment)
risk_assessment = [Interval(risk_assessment.at(i)) for i in range(len(ipomdp.states))]
print("risk=", risk_assessment, type(risk_assessment[0]))

unfolder = ObservationTraceUnfolderInterval(
    ipomdp,
    risk_assessment,
    expr_manager,
    options,
)

ura = UnfoldingIntervalRiskAssessment(stormpy_environment, unfolder, "stats/imdp", mm)
mon = monitor.Monitor(ura, None)

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
    action = input("Next Step (\\d*/r/speed) ")
