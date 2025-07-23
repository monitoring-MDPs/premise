import os
import numpy as np
import pandas as pd
from typing import Any
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import OneHotEncoder
import argparse

from premise.interval.loading import build_suo, build_suo_args_parser
from premise.interval.utils import setup_logging, logger
from premise.interval.conformence import test_monitor


def create_onehot_encoder(possible_observations, num_steps):
    possible_observations = np.array(sorted(possible_observations))
    categories = [possible_observations] * num_steps
    ohe = OneHotEncoder(
        categories=categories,
        dtype=np.int8,
        drop="first",
    )
    ohe.fit(
        X=np.array([[possible_observations[0]] * num_steps])
    )  # Fit with a dummy value
    return ohe


def prep_traces_onehot_encoder(ohe: OneHotEncoder, traces, num_steps):
    """
    Using sklearn's OneHotEncoder - good for categorical data.
    """
    # First, convert traces to a 2D array of observations
    obs_array = []
    for trace in traces:
        trace_obs = [obs for _, obs, _ in trace[:num_steps]]
        obs_array.append(trace_obs)

    obs_array = np.array(obs_array)

    # Fit and transform
    encoded_flat = ohe.transform(obs_array).toarray()

    # Reshape back to (n_traces, num_steps * n_features)
    n_traces = len(traces)
    encoded = encoded_flat.reshape(n_traces, -1)

    return encoded


def learn_regression_model(
    train_samples_amount: int,
    ohe: OneHotEncoder,
    suo,
    length: int,
    horizon: int,
    batch_size: int,
):
    logger.info("Preparing data for regression model training.")

    num_steps = length

    # Use SGDRegressor with partial_fit for incremental learning
    model = SGDRegressor()

    trained_sample_count = 0
    iteration = 0
    model_initialized = False

    while trained_sample_count < train_samples_amount:
        logger.info(f"Starting iteration {iteration} for regression model training.")
        train_samples = suo.generate_random_traces(
            [],
            length + horizon,
            min(batch_size, train_samples_amount - trained_sample_count),
        )
        trained_sample_count += len(train_samples)
        logger.info(
            f"Generated {len(train_samples)} samples in iteration {iteration}. Total trained samples: {trained_sample_count}."
        )

        # Use vectorized preprocessing - much faster than DataFrame operations
        X = prep_traces_onehot_encoder(ohe, train_samples, num_steps)

        # Vectorized label creation
        y = np.array(
            [
                1 if any(x[2] == True for x in trace[num_steps:]) else 0
                for trace in train_samples
            ]
        )

        logger.info(
            f"Prepared iteration {iteration} with {len(X)} samples ({sum(y)/len(y) * 100}% positive)."
        )

        # Use partial_fit for incremental learning (more memory efficient)
        if not model_initialized:
            model.fit(X, y)
            model_initialized = True
        else:
            model.partial_fit(X, y)

        logger.info(f"Iteration {iteration} regression model trained.")

        iteration += 1

    logger.info("Regression model training completed.")
    return model


def regression_distance(
    testing_samples, regression_risks, distance, test_weights, suo, args
):

    target_risks = test_monitor(
        suo.create_target_monitor(),
        testing_samples,
    )

    target_dist, target_all_dist = distance.distance(
        test_weights,
        {s: float(r) for s, r in target_risks.items()},
        {s: float(r) for s, r in regression_risks.items()},
        all_distances=True,
    )

    return target_risks, target_dist, target_all_dist


def reg_main(args: argparse.Namespace):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["BLIS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    setup_logging()

    logger.info(f"Running regression with args: {args}")

    suo, initial_amount, horizon = build_suo(args)

    if args.length is None:
        if initial_amount is not None:
            args.length = initial_amount
        else:
            raise ValueError(
                "Either length must be specified or initial_amount must be provided by the model."
            )

    if args.horizon is None:
        if horizon is not None:
            args.horizon = horizon
        else:
            raise ValueError(
                "Either horizon must be specified or it must be provided by the model."
            )

    all_states = suo.get_states_and_transitions()[0]

    observations = []
    for x in all_states:
        if x[1] not in observations:
            observations.append(x[1])

    one_hot_encoder = create_onehot_encoder(observations, args.length)

    model = learn_regression_model(
        args.amount,
        one_hot_encoder,
        suo,
        args.length,
        args.horizon,
        args.batch_size,
    )

    if args.dump_stats:
        np.save(
            args.dump_stats,
            {
                "args": vars(args),
                "observations": observations,
                "one_hot_encoder": one_hot_encoder,
            },  # type: ignore
            allow_pickle=True,
        )

    if args.model_path:
        np.save(args.model_path, model, allow_pickle=True)  # type: ignore


def build_learning_args_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Learning Parameters")
    group.add_argument(
        "-m", "--model-path", type=str, default=None, help="Path to store the model"
    )
    group.add_argument(
        "-a", "--amount", type=int, default=100, help="Amount of samples to generate"
    )
    group.add_argument(
        "-l", "--length", type=int, help="Length of the samples to generate"
    )
    group.add_argument(
        "-b", "--batch-size", type=int, default=5000, help="Batch size for training"
    )
    group.add_argument("--model", type=bool, default=False, help="If a model exists")
    group.add_argument("--horizon", type=int, help="Length horizon")


def reg_argsparser():
    parser = argparse.ArgumentParser(description="Learn an IMC")
    build_suo_args_parser(parser)
    build_learning_args_parser(parser)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (can be used multiple times)",
    )

    parser.add_argument("--dump-model", type=str, help="Path to dump the model to")
    parser.add_argument(
        "--dump-stats", type=str, help="Path to the file to dump stats to"
    )
    parser.add_argument(
        "-ri",
        "--run-id",
        type=int,
        default=0,
        help="Run ID to use for the experiment. Used to distinguish between different runs in the same model path.",
    )

    return parser


if __name__ == "__main__":
    parser = reg_argsparser()
    args = parser.parse_args()
    reg_main(args)
