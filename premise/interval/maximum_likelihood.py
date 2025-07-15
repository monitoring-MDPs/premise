import pickle
from typing import Optional
from stormpy import (
    SparsePomdp,
    SparseRationalPomdp,
    Rational,
    StateLabeling,
    SparseRationalModelComponents,
    SparseModelComponents,
)
import stormpy as sp
from premise.interval.interval import Samples, State
from premise.interval.utils import logger
from premise.system import SystemUnderObservation


def dict_to_pomdp(
    trans_dict: dict[tuple[State, State], float],
    init_dict: dict[State, float],
    target_label,
    use_exact=True,
) -> tuple[
    SparsePomdp | SparseRationalPomdp,
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
        components = SparseRationalModelComponents(matrix, labeling)
    else:
        components = SparseModelComponents(matrix, labeling)
    components.observability_classes = [0] + [
        o for _, o in sorted(observations.items())
    ]

    if use_exact:
        return SparseRationalPomdp(components), observation_map, state_index_map
    else:
        return SparsePomdp(components), observation_map, state_index_map


def create_mle_monitor(init_interval, interval, target_label, use_exact=True):
    pass


def maximum_likelihood_estimation(
    all_states: list[State],
    all_transitions: list[tuple[State, State]],
    samples: Samples,
    min_trans_prob: float = 0.01,
    remove_unseen_transitions: bool = True,
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
    for state, count in visit_state_count.items():
        if count > 0:
            initial_state_probabilities[state] = count / len(samples)
        else:
            initial_state_probabilities[state] = 0.0

    # Remove unseen transitions if specified
    if remove_unseen_transitions:
        initial_state_probabilities = {
            state: prob
            for state, prob in initial_state_probabilities.items()
            if prob >= min_trans_prob
        }
        transition_probabilities = {
            (src, dest): prob
            for (src, dest), prob in transition_probabilities.items()
            if prob >= min_trans_prob
        }

    return initial_state_probabilities, transition_probabilities


def mle_learning(
    suo: SystemUnderObservation,
    all_states: list[State],
    all_transitions: list[tuple[State, State]],
    iterations: int,
    initial_amount: int,
    horizon: int,
    learning_amount: int,
    testing_amount: int = 100,
    model_path: Optional[str] = None,
):
    samples = suo.generate_random_traces(
        [],
        initial_amount + horizon,
        learning_amount,
    )
    samples_per_iteration = len(samples) // iterations
    for i in range(iterations):
        logger.info(f"Iteration {i + 1}/{iterations}")

        sample_subset = samples[: samples_per_iteration * (i + 1)]
        model = maximum_likelihood_estimation(
            all_states,
            all_transitions,
            sample_subset,
        )

        with open(f"{model_path}-{i}.pickl", "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Learned transition probabilities, now testing.")

        test_samples = suo.generate_random_traces(
            [],
            initial_amount,
            testing_amount,
        )
