import argparse
import numpy

from premise.models import default_models
from premise.system import MCSystemUnderObservation, SystemUnderObservation
from premise.interval.interval import Samples, State


def premilinaries(epsilon, i_i_nl, i_i_nu, i_nl, i_nu, all_states, all_intervals):

    initial_interval = {}

    for s in all_states:
        initial_interval[s] = [epsilon, 1 - epsilon]

    strength_interval_initial = {}

    for s in all_states:
        strength_interval_initial[s] = [i_i_nl, i_i_nu]

    interval = {}

    for a, b in all_intervals:
        interval[a, b] = [epsilon, 1 - epsilon]

    strength_interval = {}

    for a, b in all_intervals:
        strength_interval[a, b] = [i_nl, i_nu]

    return initial_interval, strength_interval_initial, interval, strength_interval


def initial_interval_learning(
    all_states, samples: Samples, strength_interval_initial, initial_interval, min_width
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
                        (initial_interval[n][0] * strength_interval_initial[n][0])
                        + initial_count[n]
                    ) / (strength_interval_initial[n][0] + trace_num)
                else:
                    initial_interval[n][0] = (
                        (initial_interval[n][0] * strength_interval_initial[n][1])
                        + initial_count[n]
                    ) / (strength_interval_initial[n][1] + trace_num)

    for n in initial_count.keys():  # learns the upper bound of the initial interval
        for i in initial_interval.keys():
            if n == i:
                if any(
                    (initial_count[x] / trace_num) > initial_interval[n][1]
                    for x in initial_count.keys()
                ):
                    initial_interval[n][1] = (
                        (initial_interval[n][1] * strength_interval_initial[n][0])
                        + initial_count[n]
                    ) / (strength_interval_initial[n][0] + trace_num)
                else:
                    initial_interval[n][1] = (
                        (initial_interval[n][1] * strength_interval_initial[n][1])
                        + initial_count[n]
                    ) / (strength_interval_initial[n][1] + trace_num)

    for k in initial_interval.keys():  # Adjusting interval width
        if initial_interval[k][1] - initial_interval[k][0] < min_width:
            middle = (initial_interval[k][1] + initial_interval[k][0]) / 2
            initial_interval[k][1] = middle + (min_width / 2)
            initial_interval[k][0] = middle - (min_width / 2)

    for s in all_states:  # updates strength intervals
        strength_interval_initial[s][0] += trace_num
        strength_interval_initial[s][1] += trace_num

    for k in initial_interval.keys():
        if initial_interval[k][0] < 0:
            initial_interval[k][0] = 0.0
        elif initial_interval[k][1] > 1:
            initial_interval[k][1] = 1.0

    return initial_interval, strength_interval_initial


def interval_learning(
    all_states: list[State],
    samples: Samples,
    interval: dict[tuple[State, State], list[float]],
    strength_interval,
    min_width: float,
):

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
                    (strength_interval[i][0] * interval[i][0]) + tau_count[i]
                ) / (
                    strength_interval[i][0] + transition_count[n]
                )  # FIX THE USE OF nl and nu
            else:
                interval[i][0] = (
                    (strength_interval[i][1] * interval[i][0]) + tau_count[i]
                ) / (
                    strength_interval[i][1] + transition_count[n]
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
                    (strength_interval[i][0] * interval[i][1]) + tau_count[i]
                ) / (strength_interval[i][0] + transition_count[n])
            else:
                interval[i][1] = (
                    (strength_interval[i][1] * interval[i][1]) + tau_count[i]
                ) / (strength_interval[i][1] + transition_count[n])

    for k in interval.keys():  # Adjusting interval width
        if interval[k][1] - interval[k][0] < min_width:
            middle = (interval[k][1] + interval[k][0]) / 2
            interval[k][1] = middle + (min_width / 2)
            interval[k][0] = middle - (min_width / 2)

    for k in interval.keys():
        if interval[k][0] < 0:
            interval[k][0] = 0.0
        elif interval[k][1] > 1:
            interval[k][1] = 1.0

    for t in tau_count.keys():  # updates strength intervals
        k = t[0]
        strength_interval[t][0] += transition_count[k]
        strength_interval[t][1] += transition_count[k]

    for src, dest in tau_count.keys():
        if tau_count[src, dest] == 0:
            l, u = interval.pop((src, dest))

            # Calculate sum of interval upperbounds and lowerbounds of src
            sum_upper = sum(
                interval[(src, s)][1] for s in all_states if (src, s) in interval
            )
            sum_lower = sum(
                interval[(src, s)][0] for s in all_states if (src, s) in interval
            )

            # Normalize upper- and lowerbounds of src
            for s in all_states:
                if (src, s) in interval:
                    interval[(src, s)][0] += l * (interval[(src, s)][0] / sum_lower)
                    interval[(src, s)][1] += u * (interval[(src, s)][1] / sum_upper)

    return interval, strength_interval


def learn_IMC(
    all_states: list,
    all_transitions: list,
    samples: Samples,
    min_width: float,
    epsilon: float = 1 / 1000,
    i_i_nl: int = 5,
    i_i_nu: int = 10,
    i_nl: int = 10,
    i_nu: int = 20,
):
    initial_interval, strength_interval_initial, interval, strength_interval = (
        premilinaries(epsilon, i_i_nl, i_i_nu, i_nl, i_nu, all_states, all_transitions)
    )

    initial_interval_learning(
        all_states,
        samples,
        strength_interval_initial,
        initial_interval,
        min_width,
    )

    interval_learning(all_states, samples, interval, strength_interval, min_width)

    return initial_interval, interval


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

    if args.mc:
        model_def = default_models[args.mc]
        suo: SystemUnderObservation = MCSystemUnderObservation(model_def, args.mc)
    elif args.sim:
        suo: SystemUnderObservation = None  # type: ignore
    else:
        raise ValueError("No model specified")

    risk = suo.get_risk()
    all_states, all_transitions = suo.get_states_and_transitions()
    samples = suo.generate_random_traces([], args.length, args.amount)

    initial_interval, interval = learn_IMC(
        all_states,
        all_transitions,
        samples,
        args.interval_min_width,
        args.epsilon,
        args.initial_lower_strength,
        args.initial_upper_strength,
        args.trans_lower_strength,
        args.trans_upper_strength,
    )

    numpy.save(f"premise/examples/{suo.model_name}-initial_interval.npy", initial_interval)  # type: ignore
    numpy.save(f"premise/examples/{suo.model_name}-interval.npy", interval)  # type: ignore
