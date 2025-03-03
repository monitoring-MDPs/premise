import sys

import numpy
import models
import monitor
import trace_generator


epsilon = 1 / 1000

i_i_nl = 5  # initial lower bound of strength interval for initial distribution
i_i_nu = 10  # initial upper bound of strength interval for initial distribution

i_nl = 10  # initial lower bound of strength interval
i_nu = 20  # initial upper bound of strength interval

# all_states = []
# samples = []


def premilinaries(epsilon, i_i_nl, i_i_nu, i_nl, i_nu, all_states):

    initial_interval = {}

    for s in all_states:
        initial_interval[s] = [epsilon, 1 - epsilon]

    strenght_interval_initial = {}

    for s in all_states:
        strenght_interval_initial[s] = [i_i_nl, i_i_nu]

    interval = {}

    for a in all_states:
        for b in all_states:
            interval[a, b] = [epsilon, 1 - epsilon]

    strenght_interval = {}

    for a in all_states:
        for b in all_states:
            strenght_interval[a, b] = [i_nl, i_nu]

    return initial_interval, strenght_interval_initial, interval, strenght_interval


def initial_interval_learning(
    all_states, samples, strenght_interval_initial, initial_interval
):

    trace_num = len(samples)  # number of traces in a sample

    initial_count = {}

    for s in all_states:
        initial_count[s] = 0  # DOES IT NEED TO BE A TUPLE?

    for t in samples:
        for s in all_states:
            if s == t[0]:
                initial_count[s] += 1

    for n in initial_count.keys():  # learns the lower bound of the initial interval
        for i in initial_interval.keys():
            if n == i:
                if any(
                    (initial_count[x] / trace_num) < initial_interval[n][0]
                    for x in initial_count.keys()
                ):
                    initial_interval[n][0] = (
                        (initial_interval[n][0] * strenght_interval_initial[n][0])
                        + initial_count[n]
                    ) / (strenght_interval_initial[n][0] + trace_num)
                else:
                    initial_interval[n][0] = (
                        (initial_interval[n][0] * strenght_interval_initial[n][1])
                        + initial_count[n]
                    ) / (strenght_interval_initial[n][1] + trace_num)

    for n in initial_count.keys():  # learns the upper bound of the initial interval
        for i in initial_interval.keys():
            if n == i:
                if any(
                    (initial_count[x] / trace_num) > initial_interval[n][1]
                    for x in initial_count.keys()
                ):
                    initial_interval[n][1] = (
                        (initial_interval[n][1] * strenght_interval_initial[n][0])
                        + initial_count[n]
                    ) / (strenght_interval_initial[n][0] + trace_num)
                else:
                    initial_interval[n][1] = (
                        (initial_interval[n][1] * strenght_interval_initial[n][1])
                        + initial_count[n]
                    ) / (strenght_interval_initial[n][1] + trace_num)

    for s in all_states:  # updates strength intervals
        if strenght_interval_initial[s][0] < 100 and strenght_interval_initial[s][1] < 120:
            strenght_interval_initial[s][0] += trace_num
            strenght_interval_initial[s][1] += trace_num

    return initial_interval, strenght_interval_initial


def interval_learning(all_states, samples, interval, strenght_interval):

    trace_num = len(samples)  # number of traces in a sample
    trace_len = len(samples[0])

    transition_count = {}

    for s in all_states:
        transition_count[s] = 0

    for t in samples:
        for s in t[:-1]:
            transition_count[s] += 1

    tau_count = {}

    for a in transition_count.keys():
        for b in transition_count.keys():
            tau_count[a, b] = 0

    for x in range(trace_num):
        for y in range(trace_len - 1):
            tau_count[samples[x][y], samples[x][y + 1]] += 1

    for i in interval.keys():  # learns the lower bound of the interval
        m = i
        n = i[0]
        if transition_count[n] != 0:
            if any(
                (tau_count[n, s] / transition_count[n] < interval[n, s][0])
                for s in all_states
            ):
                interval[i][0] = (
                    (strenght_interval[m][0] * interval[i][0]) + tau_count[m]
                ) / (
                    strenght_interval[m][0] + transition_count[n]
                )  # FIX THE USE OF nl and nu
            else:
                interval[i][0] = (
                    (strenght_interval[m][1] * interval[i][0]) + tau_count[m]
                ) / (
                    strenght_interval[m][1] + transition_count[n]
                )  # FIX THE USE OF nl and nu

    for i in interval.keys():  # learns the upper bound of the interval
        n = i[0]
        m = i
        if transition_count[n] != 0:
            if any(
                (tau_count[n, s] / transition_count[n] > interval[n, s][1])
                for s in all_states
            ):
                interval[i][1] = (
                    (strenght_interval[m][0] * interval[i][1]) + tau_count[m]
                ) / (strenght_interval[m][0] + transition_count[n])
            else:
                interval[i][1] = (
                    (strenght_interval[m][1] * interval[i][1]) + tau_count[m]
                ) / (strenght_interval[m][1] + transition_count[n])

    for t in tau_count.keys():  #updates strength intervals
        if strenght_interval[t][0] < 100 and strenght_interval[t][1] < 120: 
            k = t[0]
            strenght_interval[t][0] += transition_count[k]
            strenght_interval[t][1] += transition_count[k]

    return interval, strenght_interval


# Define which model to choose from the default models
# It chooses SnL-10x10 if no model is given as an argument
model_name = sys.argv[1] if len(sys.argv) > 1 else "SnL-10x10"
model_def = models.default_models[model_name]

print(f"Learning {model_name}")

# Load the model into storm
model, risk = models.build_model_and_risk(
    model_def,
    monitor.PremiseOptions(),
)

print(model)

# Get the list of states and transitions
all_states, transtitions = models.build_state_and_transition_list(
    model, model_def.target_label
)

# Build the initial interval and interval dicts with widest ranges
initial_interval, strenght_interval_initial, interval, strenght_interval = (
    premilinaries(epsilon, i_i_nl, i_i_nu, i_nl, i_nu, all_states)
)

print("Ready for learning")

# Create the sampler
ctr = trace_generator.ConditionalTraceGenerator(
    model, target_label=model_def.target_label
)

# Sample 1000 paths of length 20
samples = [ctr.generate_random_trace([], 20) for _ in range(1000)]

print(f"Sampled {len(samples)} times")

# Learn the initial interval for all states
initial_interval_learning(
    all_states, samples, strenght_interval_initial, initial_interval
)

print("Initial interval learned")

# Learn the transition interval
interval_learning(all_states, samples, interval, strenght_interval)

print("Interval learned")

numpy.save(f"premise/examples/{model_name}-initial_interval.npy", initial_interval)  # type: ignore
numpy.save(f"premise/examples/{model_name}-interval.npy", interval)  # type: ignore
