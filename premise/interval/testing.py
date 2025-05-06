import argparse
import numpy as np
from tqdm import trange

from premise.interval.loading import (
    build_imc_loading_args_parser,
    build_suo,
    build_suo_args_parser,
    load_imc,
)
from premise.interval.conformence import test_monitor
from premise.interval.interval import create_monitor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformance checking")

    build_suo_args_parser(parser)
    build_imc_loading_args_parser(parser)

    parser.add_argument(
        "-l",
        "--sample_length",
        required=True,
        type=int,
        help="Length of the samples to generate",
    )
    parser.add_argument(
        "-s", "--samples", type=int, default=1000, help="Amount of samples to test on"
    )
    parser.add_argument(
        "-ho", "--horizon", required=True, type=int, help="The horizon to monitor on"
    )
    parser.add_argument(
        "--no-target", action="store_true", help="Do not use the target monitor"
    )
    parser.add_argument(
        "--dump", type=str, help="Path to the file to dump the model to"
    )

    parser.add_argument(
        "--dump-stats", type=str, help="Path to the file to dump stats to"
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()

    suo = build_suo(args)
    interval, initial_interval = load_imc(args)

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

    if not args.no_target:
        target_monitor = suo.create_target_monitor()

    alarms: list[bool] = []
    risks = []
    target_risks = []
    traces = []

    print("Ready for testing")

    for x in trange(args.samples):

        trace = suo.generate_random_traces([], args.sample_length + args.horizon)[0]

        traces.append(trace)
        alarms.append(any([s[2] for s in trace]))

        # Run premise on the learned model
        sub_trace = trace[: args.sample_length]
        risk = test_monitor(
            mon,
            [sub_trace],
            obs_func=lambda x: observation_map[x],
            skip_initial=True,
            with_tqdm=False,
        )[sub_trace]

        risks.append(risk)

        if not args.no_target:
            target_risk = test_monitor(
                target_monitor,
                [sub_trace],
                with_tqdm=False,
            )[sub_trace]
            target_risks.append(target_risk)

    if args.dump_stats:
        stats = {
            "samples": traces,
            "alarms": alarms,
            "risks": risks,
        }
        if not args.no_target:
            stats["target_risks"] = target_risks
        np.save(
            args.dump_stats,
            stats,  # type: ignore
        )

    print("Results:")
    print(f"Loaded {len(risks)} samples.")
    print(f"Alarms: {np.sum(alarms)} / {len(alarms)} ({100*np.mean(alarms):.2f}%)")
    print(
        f"Risks: min={np.min(risks):.4f}, max={np.max(risks):.4f}, mean={np.mean(risks):.4f}"
    )
    print(
        f"Risks: std={np.std(risks):.4f}, var={np.var(risks):.4f}, median={np.median(risks):.4f}"
    )

    # Print a trace with an alarm and without an alarm use the suo trace printer suo.trace_to_str(trace)
    print("Alarmed traces:")
    for i, trace in enumerate(traces):
        if alarms[i]:
            print(f"Trace {i}: {suo.trace_to_str(trace)}")
            print(f"Risk: {risks[i]}")
            if not args.no_target:
                print(f"Target risk: {target_risks[i]}")
                print(
                    f"True risk before horizon: {float(suo.get_risk()[trace[args.sample_length][0]])}"
                )
                print(f"True risk at end: {float(suo.get_risk()[trace[-1][0]])}")
            break
    print("Not alarmed traces:")
    for i, trace in enumerate(traces):
        if not alarms[i]:
            print(f"Trace {i}: {suo.trace_to_str(trace)}")
            print(f"Risk: {risks[i]}")
            if not args.no_target:
                print(f"Target risk: {target_risks[i]}")
                print(
                    f"True risk before horizon: {float(suo.get_risk()[trace[args.sample_length][0]])}"
                )
                print(f"True risk at end: {float(suo.get_risk()[trace[-1][0]])}")
            break
