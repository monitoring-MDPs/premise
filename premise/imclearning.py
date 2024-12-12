import scenic
import tempfile
import pathlib

from scenic.simulators.newtonian import NewtonianSimulator


def create_states(step):
    states = []

    for x in range(min, max, step):
        for y in range(min, max, step):
            for z in range(min, max, step):
                for v in range(min, max, step):
                    for k in (("close"), ("far")):
                        for j in (("collision"), ("no_collision")):
                            states.append([])
                            states[-1].append(((x, x + step), (y, y + step)))
                            states[-1].append(((z, z + step), (v, v + step)))
                            states[-1].append(k)
                            states[-1].append(j)
    return states


def obtain_traces(scenario, trace_num, trace_len):
    traces = []
    for n in range(trace_num):
        scene, _ = scenario.generate()
        simulator = NewtonianSimulator()
        simulation = simulator.simulate(scene, maxSteps=trace_len - 1)
        if (
            simulation and simulation.result
        ):  # `simulate` can return None if simulation fails
            result = simulation.result
            trace = []
            for i, state in enumerate(result.trajectory):
                egoPos, parkedCarPos = state
                state = []

                for x in range(min, max, step):
                    for y in range(min, max, step):
                        if (x < egoPos[0] <= x + step) and (y < egoPos[1] <= y + step):
                            state.append(((x, x + step), (y, y + step)))

                for z in range(min, max, step):
                    for v in range(min, max, step):
                        if (z < parkedCarPos[0] <= z + step) and (
                            v < parkedCarPos[1] <= v + step
                        ):
                            state.append(((z, z + step), (v, v + step)))

                if (
                    abs((egoPos[0]) - (parkedCarPos[0])) < closeness
                    and abs((egoPos[1]) - (parkedCarPos[1])) < closeness
                ):
                    state.append("close")
                else:
                    state.append("far")

                if (
                    abs(egoPos[0] - parkedCarPos[0]) < 4.5
                    and abs(egoPos[1] - parkedCarPos[1]) < 2
                ):
                    state.append("collision")

                else:
                    state.append("no_collision")

                trace.append(state)
            traces.append(trace)
    return traces


def create_initial_distribution(states, traces, epsilon, i_i_nl, i_i_nu, trace_num):
    initial_count = {}

    for s in states:
        initial_count[tuple(s)] = 0

    for t in traces:
        for s in states:
            if tuple(s) == t[0]:
                initial_count[tuple(s)] += 1

    initial_nl = {}
    initial_nu = {}

    for s in states:
        initial_nl[tuple(s)] = i_i_nl
        initial_nu[tuple(s)] = i_i_nu

    initial_interval = {}

    for s in states:
        initial_interval[tuple(s)] = [epsilon, 1 - epsilon]

    for n in initial_count.keys():
        for i in initial_interval.keys():
            if n == i:
                if any(
                    (initial_count[x] / trace_num) < initial_interval[n][0]
                    for x in initial_count.keys()
                ):
                    initial_interval[n][0] = (
                        (initial_interval[n][0] * initial_nl[n]) + initial_count[n]
                    ) / (initial_nl[n] + trace_num)
                else:
                    initial_interval[n][0] = (
                        (initial_interval[n][0] * initial_nu[n]) + initial_count[n]
                    ) / (initial_nu[n] + trace_num)

    for n in initial_count.keys():
        for i in initial_interval.keys():
            if n == i:
                if any(
                    (initial_count[x] / trace_num) > initial_interval[n][1]
                    for x in initial_count.keys()
                ):
                    initial_interval[n][0] = (
                        (initial_interval[n][0] * initial_nl[n]) + initial_count[n]
                    ) / (initial_nl[n] + trace_num)
                else:
                    initial_interval[n][1] = (
                        (initial_interval[n][1] * initial_nu[n]) + initial_count[n]
                    ) / (initial_nu[n] + trace_num)

    for s in states:
        initial_nl[tuple(s)] += trace_num
        initial_nu[tuple(s)] += trace_num

    return initial_interval


def create_transitions(traces, trace_len, epsilon, i_nl, i_nu):
    transition_count = {}

    for t in traces:
        for s in t[0 : trace_len - 1]:
            transition_count[tuple(s)] = 0

    for t in traces:
        for s in t[0 : trace_len - 1]:
            for k in transition_count.keys():
                if s == list(k):
                    transition_count[tuple(s)] += 1

    tau_count = {}

    for a in transition_count.keys():
        for b in transition_count.keys():
            tau_count[a, b] = 0

    for t in traces:
        for n in range(len(t) - 1):
            for a in transition_count.keys():
                for b in transition_count.keys():
                    if t[n] == list(a) and t[n + 1] == list(b):
                        tau_count[a, b] += 1

    interval = {}

    for k in tau_count.keys():
        interval[k] = [epsilon, 1 - epsilon]

    nl = {}
    nu = {}

    for k in tau_count.keys():
        nl[k] = i_nl
        nu[k] = i_nu

    for n in transition_count.keys():
        for m in tau_count.keys():
            for i in interval.keys():
                if n == m[0] and m == i:
                    if any(
                        (
                            tau_count[x] / transition_count[n] < interval[x][0]
                            and x[0] == n
                        )
                        for x in tau_count.keys()
                    ):
                        interval[i][0] = ((nl[m] * interval[i][0]) + tau_count[m]) / (
                            nl[m] + transition_count[n]
                        )
                    else:
                        interval[i][0] = ((nu[m] * interval[i][0]) + tau_count[m]) / (
                            nu[m] + transition_count[n]
                        )

    for n in transition_count.keys():
        for m in tau_count.keys():
            for i in interval.keys():
                if n == m[0] and m == i:
                    if any(
                        (
                            tau_count[x] / transition_count[n] > interval[x][1]
                            and x[0] == n
                        )
                        for x in tau_count.keys()
                    ):
                        interval[i][1] = ((nl[m] * interval[i][1]) + tau_count[m]) / (
                            nl[m] + transition_count[n]
                        )
                    else:
                        interval[i][1] = ((nu[m] * interval[i][1]) + tau_count[m]) / (
                            nu[m] + transition_count[n]
                        )

    for k in transition_count.keys():
        for t in tau_count.keys():
            if t[0] == k:
                nl[t] += transition_count[k]
                nu[t] += transition_count[k]


if __name__ == "__main__":
    scenario = scenic.scenarioFromFile(
        "../premise/examples/badlyParkedCarPullingIn.scenic",
        model="scenic.simulators.newtonian.driving_model",
        mode2D=True,
    )

    # parementers - sample set
    trace_num = 10
    trace_len = 100

    # how many rounds of learning
    rounds = 5

    # parameters - translating scenic data
    min = -300
    max = 300
    step = 150
    closeness = 20
    collision_1 = 4.5
    collision_2 = 2

    # paramenters - intervals, strength

    epsilon = 1 / 1000

    i_nl = 10  # initial lower bound of strength interval
    i_nu = 20  # initial upper bound of strength interval

    i_i_nl = 5  # initial lower bound of strength interval for initial distribution
    i_i_nu = 10
