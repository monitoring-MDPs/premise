import argparse
import pickle
from stormpy import (
    SparsePomdp,
    SparseExactPomdp,
    Rational,
    StateLabeling,
    SparseExactModelComponents,
    SparseModelComponents,
)
import stormpy as sp
import stormpy
from premise.interval.conformence import test_monitor
from premise.interval.loading import build_suo, build_suo_args_parser
from premise.models import _analyse_model
from premise.monitor import Monitor, UnfoldingRiskAssessment
from premise.interval.interval import Samples, State
from premise.interval.utils import logger, setup_logging
from premise.system import SystemUnderObservation


def dict_to_pomdp(
    trans_dict: dict[tuple[State, State], float],
    init_dict: dict[State, float],
    target_label,
    use_exact=True,
) -> tuple[
    SparsePomdp | SparseExactPomdp,
    dict[State, int],
    dict[State, int],
]:
    transitions: dict[int, dict[int, float | Rational]] = {}
    state_index_map: dict[State, int] = {}
    observations = {}
    observation_map = {}

    # Get all states
    state_index = 1

    real_states = set()
    for (s, d), _ in trans_dict.items():
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
    for d, p in sorted(
        init_dict.items(),
        key=lambda x: (x[0][0][0] if isinstance(x[0][0], tuple) else x[0][0]),
    ):
        if d not in real_states:
            continue

        if use_exact:
            prob = Rational(p)
        else:
            prob = p
        transitions[init_state][state_index_map[d]] = prob

    for (s, d), p in sorted(
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
        if use_exact:
            prob = Rational(p)
        else:
            prob = p
        transitions[s_index][d_index] = prob

    if use_exact:
        builder = sp.storage.RationalSparseMatrixBuilder(0, 0, 0, False, True)
    else:
        builder = sp.storage.SparseMatrixBuilder(0, 0, 0, False, True)

    current_row = 0
    for s, d_dict in sorted(transitions.items()):
        builder.new_row_group(current_row)
        for dest, prob in sorted(d_dict.items()):
            builder.add_next_value(current_row, dest, prob)
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

    if use_exact:
        components = SparseExactModelComponents(matrix, labeling)
    else:
        components = SparseModelComponents(matrix, labeling)
    components.observability_classes = [0] + [
        o for _, o in sorted(observations.items())
    ]

    if use_exact:
        return SparseExactPomdp(components), observation_map, state_index_map
    else:
        return SparsePomdp(components), observation_map, state_index_map


def create_mle_monitor(horizon: int, storm_model: SparsePomdp | SparseExactPomdp):
    prop_string = f'P=? [F<={horizon} "target"]'
    prop = sp.parse_properties(prop_string)[0]
    risk = _analyse_model(storm_model, prop).get_values()
    expr_manager = stormpy.ExpressionManager()
    unfolder = stormpy.pomdp.create_observation_trace_unfolder(
        storm_model, risk, expr_manager
    )
    ura = UnfoldingRiskAssessment(stormpy.Environment(), unfolder)
    mon = Monitor(ura, 1000000)
    return mon


def maximum_likelihood_estimation(
    all_states: list[State],
    all_transitions: list[tuple[State, State]],
    all_initial_states: list[State],
    samples: Samples,
):
    # Initialize counts for state visits and transitions
    visit_state_count = {state: 0 for state in all_states}
    visit_trans_count = {(src, dest): 0 for src, dest in all_transitions}

    # Count state visits and transitions from samples
    for trace in samples:
        for i in range(len(trace) - 1):
            src, dest = trace[i], trace[i + 1]
            visit_state_count[src] += 1
            visit_trans_count[src, dest] += 1

    # Compute transition probabilities
    transition_probabilities = {}
    for (src, dest), count in visit_trans_count.items():
        if visit_state_count[src] > 0:
            prob = count / visit_state_count[src]
            transition_probabilities[src, dest] = prob
        else:
            transition_probabilities[src, dest] = 0.0

    # Compute initial state probabilities
    initial_state_probabilities = {}
    for state in all_initial_states:
        count = visit_state_count.get(state, 0)
        if count > 0:
            initial_state_probabilities[state] = count / len(samples)
        else:
            initial_state_probabilities[state] = 0.0

    return initial_state_probabilities, transition_probabilities


def test_mle_monitor(
    suo: SystemUnderObservation,
    model,
    horizon: int,
    initial_length: int,
    testing_amount: int,
    use_exact: bool = True,
):
    storm_model, observation_map, state_index_map = dict_to_pomdp(
        model[1],
        model[0],
        True,
        use_exact=use_exact,
    )

    logger.info(f"Created storm model")

    mon = create_mle_monitor(horizon, storm_model)

    logger.info(f"Created monitor")

    test_samples = suo.generate_random_traces(
        [],
        initial_length,
        testing_amount,
    )

    res = test_monitor(
        mon,
        test_samples,
        lambda x: observation_map[x],
        skip_initial=True,
        with_tqdm=False,
    )

    return res


def mle_learning(
    suo: SystemUnderObservation,
    all_states: list[State],
    all_transitions: list[tuple[State, State]],
    all_initial_states: list[State],
    iterations: int,
    initial_length: int,
    horizon: int,
    learning_amount: int,
    model_path: str,
):
    samples = suo.generate_random_traces(
        [],
        initial_length + horizon,
        learning_amount,
    )
    samples_per_iteration = len(samples) // iterations

    sample_count_list = []

    for i in range(iterations):
        logger.info(f"Iteration {i + 1}/{iterations}")

        sample_subset = samples[: samples_per_iteration * (i + 1)]
        sample_count_list.append(len(sample_subset))

        model = maximum_likelihood_estimation(
            all_states,
            all_transitions,
            all_initial_states,
            sample_subset,
        )

        with open(f"{model_path}-{i}.pickl", "wb") as f:
            pickle.dump(model, f)

    return sample_count_list


def mle_learning_main(args):
    setup_logging()

    logger.info(f"Starting MLE learning with args: {args}")

    suo, initial_length, horizon = build_suo(args)
    if args.horizon is None:
        args.horizon = horizon
    if args.initial_length is None:
        args.initial_length = initial_length

    all_states, all_transitions, all_initial_states = suo.get_states_and_transitions()

    sample_count_list = mle_learning(
        suo,
        all_states,
        all_transitions,
        all_initial_states,
        args.iterations,
        args.initial_length,
        args.horizon,
        args.samples,
        args.dump_stats,
    )

    stats = {
        "args": vars(args),
        "sample_counts": sample_count_list,
        "model_paths": [f"{args.dump_model}-{i}.pickl" for i in range(args.iterations)],
    }

    with open(args.dump_stats, "wb") as f:
        pickle.dump(stats, f)

    return stats


def mle_args_parser():
    parser = argparse.ArgumentParser(description="MLE Learning")

    build_suo_args_parser(parser)

    parser.add_argument(
        "-i", "--iterations", type=int, default=10, help="Number of iterations"
    )
    parser.add_argument(
        "-il",
        "--initial-length",
        type=int,
        help="Initial length of samples",
    )
    parser.add_argument("-ho", "--horizon", type=int, help="Horizon for the MDP")
    parser.add_argument(
        "-s", "--samples", type=int, help="Number of samples to generate"
    )
    parser.add_argument(
        "-m", "--dump-model", type=str, default="model", help="Path to save the model"
    )
    parser.add_argument(
        "-stats", "--dump-stats", type=str, default="stats", help="Path to save stats"
    )
    parser.add_argument(
        "--run-id", type=int, default=0, help="Run ID for keeping track of runs"
    )
    parser.add_argument(
        "-v", "--verbose", help="Enable verbose logging", action="count", default=0
    )
    return parser


if __name__ == "__main__":
    mle_args = mle_args_parser().parse_args()
    mle_learning_main(mle_args)
    logger.info("MLE Learning completed successfully.")
