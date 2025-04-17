import argparse
import numpy

from premise.models import default_models
from premise.system import MCSystemUnderObservation, SystemUnderObservation


def premilinaries(epsilon, i_i_nl, i_i_nu, i_nl, i_nu, all_states, all_intervals):

    initial_interval = {}

    for s in all_states:
        initial_interval[s] = [epsilon, 1 - epsilon]

    strenght_interval_initial = {}

    for s in all_states:
        strenght_interval_initial[s] = [i_i_nl, i_i_nu]

    interval = {}

    for a, b in all_intervals:
        interval[a, b] = [epsilon, 1 - epsilon]

    strenght_interval = {}

    for a, b in all_intervals:
        strenght_interval[a, b] = [i_nl, i_nu]

    return initial_interval, strenght_interval_initial, interval, strenght_interval


def initial_interval_learning(
    all_states, samples, strenght_interval_initial, initial_interval, min_width
):

    trace_num = len(samples)  # number of traces in a sample

    initial_count = {}

    for s in all_states:
        initial_count[s] = 0

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

    for k in initial_interval.keys():  # Adjusting interval width
        if initial_interval[k][1] - initial_interval[k][0] < min_width:
            middle = (initial_interval[k][1] + initial_interval[k][0]) / 2
            initial_interval[k][1] = middle + (min_width/2)
            initial_interval[k][0] = middle - (min_width/2)

    for k in initial_interval.keys():
       if initial_interval[k][0] < 0:
            initial_interval[k][0] = 0.0
       if initial_interval[k][1] > 1:
            initial_interval[k][1] = 1.0

    for s in all_states:  # updates strength intervals
        strenght_interval_initial[s][0] += trace_num
        strenght_interval_initial[s][1] += trace_num
    

    return initial_interval, strenght_interval_initial


def interval_learning(all_states, samples, interval, strenght_interval, min_width):

    trace_len = len(samples[0])

    transition_count = {}

    for s in all_states:
        transition_count[s] = 0

    for t in samples:
        for s in t[:-1]:
            transition_count[s] += 1

    tau_count = {}

    for a, b in interval.keys():
        tau_count[a, b] = 0

    for s in samples:
        for y in range(trace_len - 1):
            tau_count[s[y], s[y + 1]] += 1

    for i in interval.keys():  # learns the lower bound of the interval
        n = i[0]
        if transition_count[n] != 0:
            if any(
                (tau_count[n, s] / transition_count[n] < interval[n, s][0])
                for s in all_states
                if (n, s) in interval
            ):
                interval[i][0] = (
                    (strenght_interval[i][0] * interval[i][0]) + tau_count[i]
                ) / (
                    strenght_interval[i][0] + transition_count[n]
                )  # FIX THE USE OF nl and nu
            else:
                interval[i][0] = (
                    (strenght_interval[i][1] * interval[i][0]) + tau_count[i]
                ) / (
                    strenght_interval[i][1] + transition_count[n]
                )  # FIX THE USE OF nl and nu

    for i in interval.keys():  # learns the upper bound of the interval
        n = i[0]
        if transition_count[n] != 0:
            if any(
                (tau_count[n, s] / transition_count[n] > interval[n, s][1])
                for s in all_states
                if (n, s) in interval
            ):
                interval[i][1] = (
                    (strenght_interval[i][0] * interval[i][1]) + tau_count[i]
                ) / (strenght_interval[i][0] + transition_count[n])
            else:
                interval[i][1] = (
                    (strenght_interval[i][1] * interval[i][1]) + tau_count[i]
                ) / (strenght_interval[i][1] + transition_count[n])


    for k in interval.keys():  # Adjusting interval width
        if interval[k][1] - interval[k][0] < min_width:
            middle = (interval[k][1] + interval[k][0]) / 2
            interval[k][1] = middle + (min_width/2)
            interval[k][0] = middle - (min_width/2)

    for k in interval.keys():
       if interval[k][0] < 0:
            interval[k][0] = 0.0
       if interval[k][1] > 1:
            interval[k][1] = 1.0

    for t in tau_count.keys():  # updates strength intervals
        k = t[0]
        strenght_interval[t][0] += transition_count[k]
        strenght_interval[t][1] += transition_count[k]

    return interval, strenght_interval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn an IMC")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "-mc", "--mc", type=str, help="Use the premise model with the given name"
    )
    model_group.add_argument(
        "-sim", "--sim", type=str, help="Use the simulation model with the given name"
    )
    parser.add_argument(
        "-a", "--amount", type=int, default=250, help="Amount of samples to generate"
    )
    parser.add_argument(
        "-l", "--length", type=int, default=20, help="Length of the samples to generate"
    )

    param_group = parser.add_argument_group("Internal Parameters")
    param_group.add_argument(
        "--epsilon",
        type=float,
        default=1 / 1000,
        help="Epsilon value for the initial interval",
    )
    param_group.add_argument(
        "--initial-lower-strength",
        type=int,
        default=5,
        help="Initial lower bound of strength interval for initial distribution",
    )
    param_group.add_argument(
        "--initial-upper-strength",
        type=int,
        default=10,
        help="Initial upper bound of strength interval for initial distribution",
    )
    param_group.add_argument(
        "--trans-lower-strength",
        type=int,
        default=10,
        help="Initial lower bound of strength interval",
    )
    param_group.add_argument(
        "--trans-upper-strength",
        type=int,
        default=20,
        help="Initial upper bound of strength interval",
    )
    param_group.add_argument(
        "--interval-min-width",
        type=float,
        default=0.0,
        help="Fix minimum interval width",
    )
    
    args = parser.parse_args()

    epsilon = args.epsilon

    i_i_nl = args.initial_lower_strength
    i_i_nu = args.initial_upper_strength

    i_nl = args.trans_lower_strength
    i_nu = args.trans_upper_strength

    min_width = args.interval_min_width

    if args.mc:
        model_def = default_models[args.mc]
        suo: SystemUnderObservation = MCSystemUnderObservation(model_def, args.mc)
    elif args.sim:
        suo: SystemUnderObservation = None  # type: ignore
    else:
        raise ValueError("No model specified")

    risk = suo.get_risk()
    all_states, all_transitions = suo.get_states_and_transitions()

    # Build the initial interval and interval dicts with widest ranges
    initial_interval, strenght_interval_initial, interval, strenght_interval = (
        premilinaries(epsilon, i_i_nl, i_i_nu, i_nl, i_nu, all_states, all_transitions)
    )

    print("Ready for learning")

    samples = suo.generate_random_traces([], args.length, args.amount)

    print(f"Sampled {len(samples)} times")

    # Learn the initial interval for all states
    initial_interval_learning(
        all_states, samples, strenght_interval_initial, initial_interval, min_width
    )
    
    print("Initial interval learned")
    
    # Learn the transition interval
    interval_learning(all_states, samples, interval, strenght_interval, min_width)

    print("Interval learned")

    numpy.save(f"premise/examples/{suo.model_name}-initial_interval.npy", initial_interval)  # type: ignore
    numpy.save(f"premise/examples/{suo.model_name}-interval.npy", interval)  # type: ignore

