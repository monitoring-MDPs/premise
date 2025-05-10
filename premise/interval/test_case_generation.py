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
        "-s", "--samples", type=int, default=100, help="Amount of samples to test on"
    )
    parser.add_argument(
        "-ho", "--horizon", required=True, type=int, help="The horizon to monitor on"
    )
    parser.add_argument(
        "--no-target", action="store_true", help="Do not use the target monitor"
    )
    parser.add_argument(
        "--dump", type=str, help="Path to the file to dump test traces"
    )

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

    for x in trange(args.samples):

        trace = suo.generate_random_traces([], args.sample_length + args.horizon)[0]
        traces.append(trace)

    if args.dump: 
        np.save(args.dump, traces)