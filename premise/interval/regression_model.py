import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any
from premise.monitor import Monitor
from sklearn.linear_model import LogisticRegression
from premise.interval.interval import Samples, Trace, create_monitor
import tqdm
import argparse

from premise.interval.loading import build_suo, build_suo_args_parser
from premise.models import default_models
from premise.system import MCSystemUnderObservation, SystemUnderObservation
from premise.interval.interval import Samples, State
from premise.carla.model_info import get_states_and_transitions
from premise.interval.loss import Distance, distance_measures


def learn_regression_model(train_samples, observations, testing_samples,args): 
    X = []
    y = []

    for l in train_samples:  #path
        flattened_trace = []
        for x in l[:-args.horizon]:  # Exclude the horizon length
            flattened_trace.append(x[1])  # Assuming x[1] contains the observable variables 
        X.append(flattened_trace)

        # Determine the prediction based on the horizon 
        if any(x[2] == True for x in l[-args.horizon:]):  # Look at the last h steps for error state #CHECK THE LABEL NAMES
            y.append(1)
        else:
            y.append(0)

    num_steps = args.length

    column_names = [f"Step{s}_Obs{o}" for s in range(num_steps) for o in observations]

    binary_data = []
    for trace in X:
        row = []
        for step in range(num_steps):
            obs = trace[step]
            row.extend([1 if obs == o else 0 for o in observations])
        binary_data.append(row)

    X = pd.DataFrame(binary_data, columns=column_names)

    model = LogisticRegression()
    model.fit(X, y)

    testing_traces = []

    for s in testing_samples:
        t = tuple(x[1] for x in s)
        testing_traces.append(t)


    binary_test_data = []
    for trace in testing_traces:
        row = []
        for step in range(num_steps):
            obs = trace[step]
            row.extend([1 if obs == o else 0 for o in observations])
        binary_test_data.append(row)


    X_test = pd.DataFrame(binary_test_data, columns=column_names)

    prob = model.predict_proba(X_test)

    risks = prob[:, 1]

    regression_risks = {}
    for sample, risk in zip(testing_samples, risks):
        regression_risks[tuple(sample)] = risk

    return testing_samples, regression_risks #dictionary trace + risk


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


def regression_distance(testing_samples, regression_risks, distance, test_weights, suo, args):

    target_risks = test_monitor(     #ANTONINA - CHANGE
        suo.create_target_monitor(
            args.dump_model + "target" if args.dump_model else None
        ),
        testing_samples,
    )

    target_dist, target_all_dist = distance.distance(
        test_weights,
        {s: float(r) for s, r in target_risks.items()},
        {s: float(r) for s, r in regression_risks.items()},
        all_distances=True,
    )

    return target_risks, target_dist, target_all_dist


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
        "-t", "--test_samples", type=int, default=100, help="Amount of test samples"
    )
    group.add_argument(
        "--horizon", type=int, default=20, help="Length horizon"
    )


if __name__ == "__main__":

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
        default="mae",
        help="Distance measure to use",
    )

    parser.add_argument(
        "--dump-model", type=str, help="Path to dump the model to"
        )
    parser.add_argument(
        "--dump-stats", type=str, help="Path to the file to dump stats to"
    )

    args = parser.parse_args()
    suo = build_suo(args)

    train_samples = suo.generate_random_traces([], args.length + args.horizon, args.amount)

    all_states = suo.get_states_and_transitions()[0]
    
    observations = []
    for x in all_states: 
        if x[1] not in observations: 
            observations.append(x[1])

    samples_with_prob = suo.generate_random_traces_with_prob(
            [], args.length, args.test_samples)
    
    total_prob = sum(p for _, p in samples_with_prob)
    test_weights = {s: float(p / total_prob) for s, p in samples_with_prob}
    testing_samples = [s[0] for s in samples_with_prob]

    #distance = distance_measures[args.distance](args.distance_threshold) #ANTONINA
    distance = distance_measures[args.distance]()


    testing_samples, regression_risks = learn_regression_model(train_samples, observations, testing_samples,args)
    target_risks, target_dist, target_all_dist = regression_distance(testing_samples, regression_risks, distance, test_weights, suo, args)


    if args.dump_stats:
        np.save(
            args.dump_stats,
            {
                "target_dist": target_dist,
                "target_all_dist": target_all_dist,
                "weights": {s: float(w) for s, w in test_weights.items()},
                "target_risks": {s: float(r) for s, r in target_risks.items()},
                "regression_risks": {s: float(r) for s, r in regression_risks.items()},
                "samples": testing_samples,
            },  # type: ignore
        )
    

    #target_values = [float(v) for v in target_risks.values()]
    #print(target_values)
    #print(list(regression_risks.values()))
    print(target_dist)
    #print(target_all_dist)
