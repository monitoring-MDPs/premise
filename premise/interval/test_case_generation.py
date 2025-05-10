import argparse
import numpy as np
from tqdm import trange
import os
import pickle


from premise.interval.loading import (
    build_suo,
    build_suo_args_parser,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformance checking")

    build_suo_args_parser(parser)

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
        "--dump", type=str, help="Path to the file to dump test traces"
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()

    suo = build_suo(args)
    
    traces = []

    for x in trange(args.samples):

        trace = tuple(suo.generate_random_traces([], args.sample_length + args.horizon)[0])
        traces.append(trace)

        

    
    if args.dump:
    # Construct the full path with the desired filename format
        filename = os.path.join(args.dump, f"{args.mc}_l{args.sample_length}_ho{args.horizon}.npy")

    # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the traces to the constructed filename
        #np.save(filename, traces)

        with open(filename, 'wb') as f:
            pickle.dump(traces, f)


#python -m premise.interval.test_case_generation -mc SnL-10x10 -l 15 -ho 5 --no-target --dump premise/analysis/test_sets/
#python -m premise.interval.test_case_generation -mc airportA-7-10-10 -l 25 -ho 15 --no-target --dump premise/analysis/test_sets
#python -m premise.interval.test_case_generation -mc airportA-7-50-30 -l 130 -ho 25 --no-target --dump premise/analysis/test_sets
#python -m premise.interval.test_case_generation -mc airportB-7-50-30 -l 130 -ho 25 --no-target --dump premise/analysis/test_sets
#python -m premise.interval.test_case_generation -mc evadeV-6-3 -l 20 -ho 12 --no-target --dump premise/analysis/test_sets
#python -m premise.interval.test_case_generation -mc evadeI-15 -l 20 -ho 12 --no-target --dump premise/analysis/test_sets
#python -m premise.interval.test_case_generation -mc SnLw-10x10 -sv pos -l 15 -ho 5 --no-target --dump premise/analysis/test_sets
#python -m premise.interval.test_case_generation -mc evadeV-6-3-coarse -sv start turn c_ax c_ay c_dx c_dy -l 20 -ho 12 --no-target --dump premise/analysis/test_sets
#python -m premise.interval.test_case_generation -mc airportB-7-50-30 -sv d p pobs turn -l 150 -ho 50 --no-target --dump premise/analysis/test_sets


