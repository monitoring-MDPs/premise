import argparse
from typing import Any
import numpy as np
import tqdm

from premise.interval.interval import create_monitor
from premise.system import MCSystemUnderObservation, SystemUnderObservation
from premise.monitor import Monitor
from premise.models import default_models


def test_monitor(monitor: Monitor, samples: list[list[tuple[int, int, bool]]]):
    risks: dict[list, Any] = {}
    for trace in tqdm.tqdm(samples):
        mon.initialize(0)
        observations = [t[1] for t in trace]
        for obs in observations[:-1]:
            mon.step(observation_map[obs], compute_risk=False)
        last_risk = mon.step(observation_map[observations[-1]], compute_risk=True)
        risks[trace] = last_risk
    return risks


def random_sample_monitor_test(
    suo: SystemUnderObservation, samples: list[list[tuple[int, int, bool]]]
):
    risks: dict[list, float] = {}
    for trace in tqdm.tqdm(samples):
        alarm = 0
        for _ in range(args.conformence_amount):
            new_trace = suo.generate_random_traces(
                [s[1] for s in trace], length=len(trace) + args.horizon
            )[0]
            alarm += any([s[2] for s in new_trace])
        risks[trace] = alarm / args.conformence_amount
    return risks


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
        "-l",
        "--sample_length",
        required=True,
        type=int,
        help="Length of the samples to generate",
    )
    parser.add_argument(
        "--conformence-amount",
        type=int,
        default=500,
        help="Amount of extensions to generate for a sample",
    )
    parser.add_argument(
        "--horizon", required=True, type=int, help="The horizon to monitor on"
    )
    parser.add_argument(
        "--dump", type=str, help="Path to the file to dump the model to"
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()

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
        args.dump,
        args.verbose,
    )

    # Run premise on the learned model
    samples = suo.generate_random_traces([], args.sample_length, 50)
    monitored_risks = test_monitor(
        mon, [[(s[0], observation_map[s[1]], s[2]) for s in t] for t in samples]
    )

    print("Monitoring learned model done")

    # Run premise on the true model
    target_risks = test_monitor(suo.create_target_monitor(), samples)

    print("Monitoring done")

    # Run the conformance test
    sampled_risks = random_sample_monitor_test(suo, samples)

    # Check correctness of the monitor
    res = []
    for s in samples:
        res.append((monitored_risks[s], sampled_risks[s], target_risks[s], s))
        if args.verbose > 0:
            print(
                f"mon:{monitored_risks[s]} real:{sampled_risks[s]} real mon:{float(target_risks[s])} \n({[(s[0], s[1], s[2]) for s in s]})\n"
            )

    # Print statistics
    print("Results:")
    print(f"Avg diff to sampling: {np.mean([abs(r[0] - r[1]) for r in res])}")
    print(f"Avg diff to premise: {np.mean([abs(r[0] - float(r[2])) for r in res])}")
    print(f"Best trace: {min(res, key=lambda x: abs(x[0] - x[1]))}")
    print(f"Worst trace: {max(res, key=lambda x: abs(x[0] - x[1]))}")
