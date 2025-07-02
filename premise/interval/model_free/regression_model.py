import numpy as np
import pandas as pd
from typing import Any
from premise.interval.conformence import random_sample_monitor_test
from premise.monitor import Monitor
from sklearn.linear_model import LogisticRegression
from premise.interval.interval import Samples, Trace
import tqdm
import argparse

from premise.interval.loading import build_suo, build_suo_args_parser
from premise.interval.interval import Samples
from premise.interval.loss import distance_measures


def prep_trace_for_regression(trace, observations):
    row = []
    for _, obs, _ in trace:
        row.extend([1 if obs == o else 0 for o in observations])
    return row


def learn_regression_model(train_samples, observations, testing_samples, args):
    num_steps = args.length

    column_names = [f"Step{s}_Obs{o}" for s in range(num_steps) for o in observations]

    binary_data = []
    y = []
    for trace in train_samples:
        sub_trace = trace[:num_steps]
        row = prep_trace_for_regression(sub_trace, observations)
        binary_data.append(row)

        y.append(1 if any(x[2] == True for x in sub_trace) else 0)

    X = pd.DataFrame(binary_data, columns=column_names)

    model = LogisticRegression(n_jobs=1)
    model.fit(X, y)

    binary_test_data = []
    for trace in testing_samples:
        sub_trace = trace[:num_steps]
        row = prep_trace_for_regression(sub_trace, observations)
        binary_test_data.append(row)

    X_test = pd.DataFrame(binary_test_data, columns=column_names)

    prob = model.predict_proba(X_test)

    risks = prob[:, 1]

    regression_risks = {}
    for sample, risk in zip(testing_samples, risks):
        regression_risks[tuple(sample)] = risk

    return testing_samples, regression_risks, model  # dictionary trace + risk


def test_monitor(
    mon: Monitor,
    samples: Samples,
    obs_func=lambda x: x,
    skip_initial=False,
    with_tqdm=True,
):
    risks: dict[Trace, Any] = {}
    it = tqdm.tqdm(samples) if with_tqdm else samples
    for trace in it:
        observations = [t[1] for t in trace]

        if skip_initial:
            mon.initialize(0)
        else:
            mon.initialize(obs_func(observations[0]))

        for obs in observations[0 if skip_initial else 1 : -1]:
            mon.step(obs_func(obs), compute_risk=False)

        last_risk = mon.step(obs_func(observations[-1]), compute_risk=True)
        risks[tuple(trace)] = last_risk
    return risks


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

    train_samples = suo.generate_random_traces(
        [], args.length + args.horizon, args.amount
    )

    all_states = suo.get_states_and_transitions()[0]

    observations = []
    for x in all_states:
        if x[1] not in observations:
            observations.append(x[1])

    samples_with_prob = suo.generate_random_traces_with_prob(
        [], args.length, args.test_samples
    )

    total_prob = sum(p for _, p in samples_with_prob)
    test_weights = {s: float(p / total_prob) for s, p in samples_with_prob}
    testing_samples = [s[0] for s in samples_with_prob]

    distance = distance_measures[args.distance]()

    testing_samples, regression_risks, model = learn_regression_model(
        train_samples, observations, testing_samples, args
    )
    target_risks, target_dist, target_all_dist = regression_distance(
        testing_samples, regression_risks, distance, test_weights, suo, args
    )

    sampled_risks = random_sample_monitor_test(
        suo, testing_samples, args.horizon, args.conformence_amount
    )

    if args.dump_stats:
        np.save(
            args.dump_stats,
            {
                "target_dist": target_dist,
                "target_all_dist": target_all_dist,
                "weights": {s: float(w) for s, w in test_weights.items()},
                "target_risks": {s: float(r) for s, r in target_risks.items()},
                "regression_risks": {s: float(r) for s, r in regression_risks.items()},
                "sampled_risks": sampled_risks,
                "samples": testing_samples,
                "args": vars(args),
                "observations": observations,
            },  # type: ignore
        )

    if args.model_path:
        np.save(args.model_path, model)


def build_learning_args_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Learning Parameters")
    group.add_argument(
        "-m", "--model-path", type=str, default=None, help="Path to store the model"
    )
    group.add_argument(
        "-a", "--amount", type=int, default=100, help="Amount of samples to generate"
    )
    group.add_argument(
        "-l", "--length", type=int, default=20, help="Length of the samples to generate"
    )
    group.add_argument(
        "-t", "--test_samples", type=int, default=50, help="Amount of test samples"
    )
    group.add_argument("--model", type=bool, default=False, help="If a model exists")
    group.add_argument("--horizon", type=int, default=20, help="Length horizon")


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
    parser.add_argument(
        "-d",
        "--distance",
        choices=distance_measures.keys(),
        default="umse",
        help="Distance measure to use",
    )
    parser.add_argument(
        "-ca",
        "--conformence-amount",
        type=int,
        default=500,
        help="Amount of extensions to generate for a sample",
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
