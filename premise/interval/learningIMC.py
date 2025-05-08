import argparse
import numpy

from premise.interval.loading import build_suo, build_suo_args_parser
from premise.interval.interval import Samples, State


def premilinaries(
    epsilon, i_i_nl, i_i_nu, i_nl, i_nu, all_states, all_intervals, initial_states
):

    initial_interval = {}

    for s in initial_states:
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
    all_states,
    samples: Samples,
    strength_interval_initial,
    initial_interval,
    min_width,
):

    trace_num = len(samples)  # number of traces in a sample

    initial_count = {}

    for s in all_states:
        initial_count[s] = 0

    for t in samples:
        initial_count[t[0]] += 1

    for n in initial_interval.keys():  # learns the lower bound of the initial interval
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

    for n in initial_interval.keys():  # learns the upper bound of the initial interval
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
    remove_unseen_transitions: bool = True,
    min_trans_prob: float = 0.01,
):

    visit_state_count = {}

    for s in all_states:
        visit_state_count[s] = 0

    for t in samples:
        for s in t[:-1]:
            visit_state_count[s] += 1

    one_transition: dict[State, bool | State] = {s: False for s in all_states}
    visit_trans_count = {}
    for a, b in interval.keys():
        visit_trans_count[a, b] = 0
        if one_transition[a] == False:
            one_transition[a] = b
        elif one_transition[a] != True and one_transition[a] != False:
            one_transition[a] = True

    for s in all_states:
        if not isinstance(one_transition[s], bool):
            interval[s, one_transition[s]] = [1.0, 1.0]  # type: ignore

    for s in samples:
        for y in range(len(s) - 1):
            visit_trans_count[s[y], s[y + 1]] += 1

    for i in interval.keys():  # learns the lower bound of the interval
        n = i[0]
        if visit_state_count[n] != 0:
            if any(
                (visit_trans_count[n, s] / visit_state_count[n] < interval[n, s][0])
                for s in all_states
                if (n, s) in interval
            ):
                interval[i][0] = (
                    (strength_interval[i][0] * interval[i][0]) + visit_trans_count[i]
                ) / (
                    strength_interval[i][0] + visit_state_count[n]
                )  # FIX THE USE OF nl and nu
            else:
                interval[i][0] = (
                    (strength_interval[i][1] * interval[i][0]) + visit_trans_count[i]
                ) / (
                    strength_interval[i][1] + visit_state_count[n]
                )  # FIX THE USE OF nl and nu

    for i in interval.keys():  # learns the upper bound of the interval
        n = i[0]
        if visit_state_count[n] != 0:
            if any(
                (visit_trans_count[n, s] / visit_state_count[n] > interval[n, s][1])
                for s in all_states
                if (n, s) in interval
            ):
                interval[i][1] = (
                    (strength_interval[i][0] * interval[i][1]) + visit_trans_count[i]
                ) / (strength_interval[i][0] + visit_state_count[n])
            else:
                interval[i][1] = (
                    (strength_interval[i][1] * interval[i][1]) + visit_trans_count[i]
                ) / (strength_interval[i][1] + visit_state_count[n])

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

    for t in visit_trans_count.keys():  # updates strength intervals
        k = t[0]
        strength_interval[t][0] += visit_state_count[k]
        strength_interval[t][1] += visit_state_count[k]

    if remove_unseen_transitions:
        to_remove = {}
        for src, dest in visit_trans_count.keys():
            if (
                visit_trans_count[src, dest] == 0
                and visit_state_count[src] != 0
                and 1 / visit_state_count[src]
                < min_trans_prob  # TODO: I am not sure of this condition
            ):
                if src not in to_remove:
                    to_remove[src] = []

                to_remove[src].append(dest)

        for src, dests in to_remove.items():
            remove_prob = [0.0, 0.0]
            for dest in dests:
                l, u = interval.pop((src, dest))
                remove_prob[0] += l
                remove_prob[1] += u

            if remove_prob[0] > 0.0 or remove_prob[1] > 0.0:
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
                        interval[(src, s)][0] += remove_prob[0] * (
                            interval[(src, s)][0] / sum_lower
                        )
                        interval[(src, s)][1] += remove_prob[1] * (
                            interval[(src, s)][1] / sum_upper
                        )

    return interval, strength_interval


def learn_IMC(
    all_states: list,
    all_transitions: list,
    initial_states: list,
    samples: Samples,
    min_width: float,
    epsilon: float = 1 / 1000,
    i_i_nl: int = 5,
    i_i_nu: int = 10,
    i_nl: int = 10,
    i_nu: int = 20,
    remove_unseen_transitions: bool = True,
    min_trans_prob: float = 0.01,
):
    initial_interval, strength_interval_initial, interval, strength_interval = (
        premilinaries(
            epsilon,
            i_i_nl,
            i_i_nu,
            i_nl,
            i_nu,
            all_states,
            all_transitions,
            initial_states,
        )
    )

    initial_interval_learning(
        all_states,
        samples,
        strength_interval_initial,
        initial_interval,
        min_width,
    )

    interval_learning(
        all_states,
        samples,
        interval,
        strength_interval,
        min_width,
        remove_unseen_transitions,
        min_trans_prob,
    )

    return initial_interval, interval


def build_learning_args_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Learning Parameters")
    group.add_argument(
        "-m", "--model-path", type=str, default=None, help="Path to store the model"
    )
    group.add_argument(
        "-a", "--amount", type=int, default=1000, help="Amount of samples to generate"
    )
    group.add_argument(
        "-l", "--length", type=int, default=20, help="Length of the samples to generate"
    )
    group.add_argument(
        "-t",
        "--existing-transitions",
        action="store_true",
        help="Use only real transitions as defined by the model",
    )
    group.add_argument(
        "--min-trans-prob",
        type=float,
        default=None,
        help="Minimum transition probability assumed of a transition",
    )


def build_learning_params_args_parser(parser: argparse.ArgumentParser):
    param_group = parser.add_argument_group("Internal Learning Parameters")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn an IMC")
    build_suo_args_parser(parser)
    build_learning_args_parser(parser)
    build_learning_params_args_parser(parser)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (can be used multiple times)",
    )

    args = parser.parse_args()

    suo = build_suo(args)

    all_states, all_transitions, initial_states = suo.get_states_and_transitions(
        all_transitions=not args.existing_transitions
    )
    samples = suo.generate_random_traces([], args.length, args.amount)

    initial_interval, interval = learn_IMC(
        all_states,
        all_transitions,
        initial_states,
        samples,
        args.interval_min_width,
        args.epsilon,
        args.initial_lower_strength,
        args.initial_upper_strength,
        args.trans_lower_strength,
        args.trans_upper_strength,
        remove_unseen_transitions=args.min_trans_prob is not None,
        min_trans_prob=args.min_trans_prob,
    )

    model_path = args.model_path
    if model_path is None:
        model_path = f"out/{suo.model_name}"
    numpy.save(f"{model_path}-initial_interval.npy", initial_interval)  # type: ignore
    numpy.save(f"{model_path}-interval.npy", interval)  # type: ignore
