import argparse
from typing import Any
import numpy as np
import tqdm

from premise.interval.interval import Samples, Trace, create_monitor
from premise.system import MCSystemUnderObservation, SystemUnderObservation
from premise.monitor import Monitor
from premise.models import default_models
from premise.interval.loss import distance_measures


def test_monitor(
    mon: Monitor, samples: Samples, obs_func=lambda x: x, skip_initial=False
):
    risks: dict[Trace, Any] = {}
    for trace in tqdm.tqdm(samples):
        observations = [t[1] for t in trace]

        if skip_initial:
            mon.initialize(0)
        else:
            mon.initialize(obs_func(observations[0]))

        for obs in observations[0 if skip_initial else 1 : -1]:
            mon.step(obs_func(obs), compute_risk=False)

        last_risk = mon.step(obs_func(observations[-1]), compute_risk=True)
        risks[trace] = last_risk
    return risks


def random_sample_monitor_test(suo: SystemUnderObservation, samples: Samples):
    risks: dict[Trace, float] = {}
    for trace in tqdm.tqdm(samples):
        alarm = 0
        for _ in range(args.conformence_amount):
            new_trace = suo.generate_random_traces(
                [s[1] for s in trace], length=len(trace) + args.horizon
            )[0]
            alarm += any([s[2] for s in new_trace])
        risks[trace] = alarm / args.conformence_amount
    return risks


def main(args: argparse.Namespace):
    if args.mc:
        model_def = default_models[args.mc]
        model_def.risk_property = (
            f'Pmax=? [F<={args.horizon} "{model_def.target_label}" ]'
        )
        suo: SystemUnderObservation = MCSystemUnderObservation(model_def, args.mc)

        if args.verbose > 1:
            for i, r in enumerate(suo.get_risk()):
                print(f"{suo._model.state_valuations.get_string(i)}: {float(r)}")
    elif args.sim:
        suo: SystemUnderObservation = None  # type: ignore
    else:
        raise ValueError("No model specified")

    distance = distance_measures[args.distance].distance

    # Load learned model
    interval = np.load(args.trans_path, allow_pickle=True)[()]
    initial_interval = np.load(args.init_path, allow_pickle=True)[()]

    print("Ready for conformance checking")

    # Build the premise monitor on the learned model
    mon, observation_map, unfolder, ipomdp = create_monitor(
        interval,
        initial_interval,
        "min",
        True,
        args.horizon,
    )

    # Run premise on the learned model
    samples_with_prob = suo.generate_random_traces_with_prob([], args.sample_length, args.conformence_amount)  # type: ignore
    total_prob = sum(p for _, p in samples_with_prob)
    weights = {s: float(p / total_prob) for s, p in samples_with_prob}
    samples = [s[0] for s in samples_with_prob]
    monitored_risks = test_monitor(
        mon,
        samples,
        obs_func=lambda x: observation_map[x],
        skip_initial=True,
    )
    # Run premise on the true model
    target_risks = test_monitor(suo.create_target_monitor(), samples)

    # Run the conformance test
    sampled_risks = random_sample_monitor_test(suo, samples)

    target_dist, target_all_dist = distance(
        weights,
        {s: float(r) for s, r in target_risks.items()},
        {s: float(r) for s, r in monitored_risks.items()},
        all_distances=True,
    )

    sample_dist, sample_all_dist = distance(
        weights,
        {s: float(r) for s, r in target_risks.items()},
        sampled_risks,
        all_distances=True,
    )

    # Save statistics as .npy
    if args.dump_stats:
        np.save(
            args.dump_stats,
            {
                "target_dist": target_dist,
                "sample_dist": sample_dist,
                "target_all_dist": target_all_dist,
                "sample_all_dist": sample_all_dist,
                "weights": {s: float(w) for s, w in weights.items()},
                "target_risks": {s: float(r) for s, r in target_risks.items()},
                "monitored_risks": {s: float(r) for s, r in monitored_risks.items()},
                "sampled_risks": sampled_risks,
                "samples": samples,
            },  # type: ignore
        )

    # Print statistics
    print("Results:")
    print(f"Distance to sampling: {sample_dist}")
    print(f"Distance to premise: {target_dist}")
    print(f"Best trace: {min(target_all_dist, key=lambda x: x[1][1])}")
    print(f"Worst trace: {max(target_all_dist, key=lambda x: x[1][1])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformance checking")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "-mc", "--mc", type=str, help="Use the premise model with the given name"
    )
    model_group.add_argument(
        "-sim", "--sim", type=str, help="Use the simulation model with the given name"
    )

    parser.add_argument(
        "-t",
        "--trans_path",
        required=True,
        type=str,
        help="Path to the transition dictionary",
    )
    parser.add_argument(
        "-i",
        "--init_path",
        required=True,
        type=str,
        help="Path to the initial state dictionary",
    )
    parser.add_argument(
        "-d",
        "--distance",
        choices=distance_measures.keys(),
        default="mae",
        help="Distance measure to use",
    )
    parser.add_argument(
        "-l",
        "--sample_length",
        required=True,
        type=int,
        help="Length of the samples to generate",
    )
    parser.add_argument(
        "-a",
        "--conformence-amount",
        type=int,
        default=500,
        help="Amount of extensions to generate for a sample",
    )
    parser.add_argument(
        "-ho", "--horizon", required=True, type=int, help="The horizon to monitor on"
    )
    parser.add_argument(
        "--dump-stats", type=str, help="Path to the file to dump stats to"
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()
    main(args)
