import pickle
from typing import Optional
import numpy as np
from premise.interval.interval import Samples, State
from premise.interval.utils import logger
from premise.system import SystemUnderObservation


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
