from datetime import datetime
import logging
from multiprocessing import set_start_method
import os
from pathlib import Path
import random
import time
import traceback

import stormpy as sp
import argparse

from tqdm import tqdm
import monitoring
from trace_generator import pre_generate_traces


class Benchmark:
    """
    Container for benchmarks.
    """

    def __init__(self, name, modelpath, constants, risk_def):
        self.name = name
        self.modelpath = modelpath
        self.constants = constants
        self.risk_def = risk_def


# Benchmarks
benchmarks = [
    Benchmark(
        "airportB-3-300-30",
        "premise/examples/airportB-3.nm",
        "DMAX=300,PMAX=30",
        'Pmax=? [F "crash"]',
    ),
    Benchmark(
        "evadeI-10",
        "premise/examples/hidden-incentive.nm",
        "N=10",
        'Pmax=? [F<=12 "crash"]',
    ),
    Benchmark(
        "evadeV-9-3",
        "premise/examples/evade-monitoring.nm",
        "N=9,RADIUS=3",
        'Pmax=? [F<=12 "crash"]',
    ),
    Benchmark(
        "refuelA-50-80",
        "premise/examples/refuel.nm",
        "N=50,ENERGY=80",
        'Pmax=? [F<=20 "empty"]',
    ),
]


def create_custom_str(exact, conditional_mode, threshold):
    parts = []
    parts.append("exact" if exact else "float")
    parts.append(conditional_mode)
    if threshold is not None:
        parts.append(f"thresh={threshold}")
    return "-".join(parts)


def make_environment(exact: bool, mode: str, threshold: bool | None) -> sp.Environment:
    env = sp.Environment()
    if exact:
        env.solver_environment.set_force_exact()
    if threshold is None:
        env.model_checker_environment.conditional.precision = sp.Rational(1e-2)
    if mode == "restart":
        env.model_checker_environment.conditional.algorithm = (
            sp.ConditionalAlgorithmSetting.restart
        )
    elif mode == "bisection":
        env.model_checker_environment.conditional.algorithm = (
            sp.ConditionalAlgorithmSetting.bisection
        )
    return env


configurations = [
    monitoring.UnfoldingOptions(
        env=make_environment(exact, conditional_mode, threshold),
        exact_arithmetic=exact,
        use_rejection_sampling=(conditional_mode == "rejection"),
        conditional_method=conditional_mode,
        model_checking_method=None,
        threshold=threshold,
        custom_str=create_custom_str(exact, conditional_mode, threshold),
    )
    for exact in [False, True]
    # for force_exact in [False, True]
    for conditional_mode in ["bisection", "rejection", "restart"]
    for threshold in [0.05, None]
    # if not (not exact and force_exact)
]

fast_configurations = [
    monitoring.UnfoldingOptions(
        env=make_environment(exact, conditional_mode, threshold),
        exact_arithmetic=exact,
        use_rejection_sampling=(conditional_mode == "rejection"),
        conditional_method=conditional_mode,
        model_checking_method=None,
        threshold=threshold,
        custom_str=create_custom_str(exact, conditional_mode, threshold),
    )
    for exact in [True]
    for conditional_mode in ["bisection", "rejection", "restart"]
    for threshold in [0.05]
]


def run_benchmark_with_config(args):
    benchmark, config, seed, trace_length, promptness_deadline, stats_path = args

    # print(f"Running {benchmark.name} with {str(config)}")

    # Set logger to output to file in stats_path
    log_file = stats_path / "logs" / f"{benchmark.name}-{str(config._numstr)}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
        force=True,
    )

    monitoring.run_monitor(
        benchmark.modelpath,
        benchmark.risk_def,
        benchmark.constants,
        trace_length,
        config,
        verbose=False,
        promptness_deadline=promptness_deadline,
        simulator_seed=seed,
        model_id=benchmark.name,
        stats_path=stats_path,
    )


if __name__ == "__main__":
    # Wait for termination, never crash.
    sp.set_settings(["--signal-timeout", "100000"])
    parser = argparse.ArgumentParser(description="Run experiments with premise.")
    parser.add_argument(
        "--number-traces", default=10, type=int, help="How many traces to run"
    )
    parser.add_argument(
        "--trace-length", default=250, type=int, help="How long should the traces be?"
    )
    parser.add_argument(
        "--promptness-deadline",
        default=1000,
        type=int,
        help="How long may one iteration take at most?",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable extra output")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set a random seed for reproducible experiments",
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick smoke test with only one configuration to verify setup.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a quick test with only the fast configurations to verify setup.",
    )
    args = parser.parse_args()

    nr_traces = args.number_traces
    trace_length = args.trace_length
    promptness_deadline = args.promptness_deadline  # in ms

    random.seed(args.seed)
    seeds = [random.getrandbits(64) for _ in range(args.number_traces)]

    stats_path = Path(args.results_folder)
    stats_path.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        benchmarks = [benchmarks[0]]
        configurations = [configurations[0]]
        print(
            "Running in smoke test mode with only one benchmark and one configuration."
        )

    if args.fast:
        configurations = fast_configurations
        print("Running in fast mode with only fast configurations.")

    bar = tqdm(total=len(benchmarks) * len(configurations))
    n = 0
    for benchmark in benchmarks:
        pre_generate_traces(
            benchmark.name,
            benchmark.modelpath,
            benchmark.constants,
            benchmark.risk_def,
            monitoring.UnfoldingOptions(sp.Environment()),
            trace_length,
            stats_path,
            seeds,
        )

        for config in configurations:
            # bar.set_description(f"Running {benchmark.name} with {str(config)}")
            monitoring.run_monitor(
                benchmark.modelpath,
                benchmark.risk_def,
                benchmark.constants,
                trace_length,
                config,
                verbose=args.verbose,
                promptness_deadline=promptness_deadline,
                simulator_seed=seeds,
                model_id=benchmark.name,
                stats_path=stats_path,
            )
            n += 1
            bar.update(n)
    print("All experiments completed.")
