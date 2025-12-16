#!/usr/bin/env python3
"""
Simple benchmark script for Storm model checking on BN-benchmarks-mdp models.
"""

import argparse
import json
import os
from pprint import pprint
from queue import Empty
from random import shuffle
import time
from pathlib import Path
from multiprocessing import Process, Queue, get_context
from typing import cast
from unittest import result
from setproctitle import setproctitle
from tqdm import tqdm
import stormpy as sp

BRP_PROPERTIES = [
    'F "success" || F "retries_MAX_min_1"',
    'F "success" || F "retries_MAX_min_3"',
]

CROWDS_PROPERTIES = [
    'F "observe0Greater1" || F "observeIGreater1"',
    'F "observeIGreater1" || F "observe0Greater1"',
]

WLAN_PROPERTIES = [
    'F "collision4" || F "collision2"',
]

# Define properties for each model
PROPERTIES = {
    "alarm": ['F "HYPOVOLEMIA1" || F "ANAPHYLAXIS1"', 'F "PAP0" || F "HR0"'],
    "andes": [
        'F "SNode_1251" || F ("SNode_111" | "SNode_201" | "FORCE601")',
        'F "SNode_1251" || F "SNode_80"',
    ],
    # "asia": ['F "dysp1" || F "bronc1"', 'F "dysp1" || F "lung0"'],
    "barley": [
        'F "protein4" || F "ngodn7"',
        'F "protein4" || F "jordinf1"',
    ],
    # "cancer": ['F "Xray1" || F "Smoker1"', 'F "Xray0" || F "Cancer0"'],
    "child": [
        'F ("XrayReport1" | "XrayReport2" | "XrayReport3" | "XrayReport4") || F "LVHreport1"',
        'F "Disease4" || F "CO21"',
    ],
    # "earthquake": ['F "MaryCalls1" || F "Earthquake1"', 'F "Alarm0" || F "JohnCalls0"'],
    "hailfinder": [
        'F "R5Fcst2" || F ("Date2" | "SubjVertMo1" | "SatContMoist0")',
        'F "R5Fcst0" || F "ScenRelAMIns2"',
    ],
    "hepar2": [
        'F "diabetes1" || F ("age1" | "age2" | "age3")',
        'F "hospital0" || F "hbeag0"',
    ],
    "insurance": [
        'F ("Accident1" | "Accident=2" | "Accident=3") || F "SeniorTrain1"',
        'F ("Accident1" | "Accident=2" | "Accident=3") || F "Theft0"',
    ],
    "pathfinder": [
        'F "F533" || F "Fault15"',
        'F "F533" || F "Fault15"',
    ],
    "sachs": [
        'F "P382" || F ("Jnk1"   | "Jnk2")',
        'F "P382" || F "PIP22"',
    ],
    # "survey": ['F ("T1" | "T2") || F "E1"', 'F "T2" || F "O1"'],
    "win95pts": [
        'F "PrtStatMem1" || F "DataFile1"',
        'F "PrtData0" || F "PrtThread1"',
    ],
    "water": [
        'F "CKNI_12_452" || F "CKNI_12_000"',
        'F "CBODD_12_150" || F "CNON_12_453"',
    ],
    "brp-N=16-MAX=8": BRP_PROPERTIES,
    "brp-N=32-MAX=16": BRP_PROPERTIES,
    "crowds_3-5": CROWDS_PROPERTIES,
    "crowds_5-5": CROWDS_PROPERTIES,
    # "crowds_10-5": CROWDS_PROPERTIES,
    "wlan": WLAN_PROPERTIES,
}

# Threshold for bounded properties (can be changed)
THRESHOLD = 0.5


def load_model(drn_file, exact=True):
    """Load a model from a DRN file."""
    opts = sp.DirectEncodingParserOptions()
    opts.build_choice_labels = True
    if exact:
        return sp._core._build_sparse_exact_model_from_drn(str(drn_file), opts)
    else:
        return sp.build_model_from_drn(str(drn_file), opts)


def _setup_environment(method, exact_mode):
    """Set up Storm environment with the specified method and arithmetic mode."""
    env = sp.Environment()
    env.solver_environment.minmax_solver_environment.precision = sp.Rational(1e-6)

    if method == "bisection":
        env.model_checker_environment.conditional_algorithm = (
            sp.ConditionalAlgorithmSetting.bisection
        )
        env.solver_environment.minmax_solver_environment.method = (
            sp.MinMaxMethod.value_iteration
        )
    elif method == "bisection-advanced":
        env.model_checker_environment.conditional_algorithm = (
            sp.ConditionalAlgorithmSetting.bisection_advanced
        )
        env.solver_environment.minmax_solver_environment.method = (
            sp.MinMaxMethod.value_iteration
        )
    elif method == "restart":
        env.model_checker_environment.conditional_algorithm = (
            sp.ConditionalAlgorithmSetting.restart
        )
    elif method == "pi":
        env.model_checker_environment.conditional_algorithm = (
            sp.ConditionalAlgorithmSetting.policy_iteration
        )

    if exact_mode == "force-exact":
        env.solver_environment.set_force_exact()

    return env


def _compute_marginal(model, path_formula):
    """Compute marginal probability for the second disjunct of the path formula.

    Assumes `path_formula` has the form '... || ...'. Uses the right-hand side.
    """
    try:
        rhs = path_formula.split("||")[1]
        marginal_prop = sp.parse_properties(f"Pmax=? [{rhs}]")[0]
        marginal_res = sp.model_checking(model, marginal_prop, only_initial_states=True)
        return float(marginal_res.at(model.initial_states[0]))
    except Exception:
        return None


def _run_single_check(
    drn_file_str, exact_mode, method, path_formula, query_type, threshold
):
    """Worker task: run a single property/method check.

    This runs as a separate process in the pool.
    Returns a single result dict.
    """
    try:
        drn_file = Path(drn_file_str)
        model_name = drn_file.stem

        # Load model
        model = load_model(drn_file, exact="exact" in exact_mode)

        # Compute marginal
        marginal = _compute_marginal(model, path_formula)

        # Build property string
        if query_type == "quantitative":
            prop_str = f"Pmax=? [{path_formula}]"
        else:  # bounded
            prop_str = f"Pmax>={threshold} [{path_formula}]"

        # Skip unsupported combination
        if method == "bisection-advanced" and exact_mode == "force-exact":
            return None

        # Set up environment and run model checking
        setproctitle(f"bn-benchmark:{model_name}:{exact_mode}:{method}:{query_type}")
        prop = sp.parse_properties(prop_str)[0]
        env = _setup_environment(method, exact_mode)

        start = time.time()
        result = sp.model_checking(
            model, prop, environment=env, only_initial_states=True
        )
        check_time = time.time() - start
        value = float(result.at(model.initial_states[0]))

        result_dict = _create_result(
            model_name,
            exact_mode,
            method,
            path_formula,
            query_type,
            threshold,
            timeout=False,
            value=value,
            marginal=marginal,
            states=model.nr_states,
            check_time=check_time,
        )

        if query_type == "bounded":
            result_dict["threshold"] = threshold

        return result_dict

    except Exception as e:
        # Return error result
        result_dict = _create_result(
            Path(drn_file_str).stem,
            exact_mode,
            method,
            path_formula,
            query_type,
            threshold,
            error=str(e),
        )
        if query_type == "bounded":
            result_dict["threshold"] = threshold

        return result_dict


def _worker_task(task, out_q: Queue):
    """Worker task to run in a separate process."""
    setproctitle(
        f"python benchmark.py {Path(task[0]).stem}:{task[1]}:{task[2]}:{task[3]}:{task[4]}"
    )
    try:
        res = _run_single_check(*task)
        if res is None:
            out_q.put(
                (
                    "skip",
                    f"Skipping unsupported combination: {task[1]}, {task[2]}",
                )
            )
        else:
            out_q.put(("ok", res))
    except Exception as e:
        out_q.put(("error", str(e)))


def _run_in_parrallel(tasks, cores, timeout, output_file):
    ctx = get_context("spawn")
    pending = list(tasks)
    running: dict[int, tuple[tuple, Process, Queue, float]] = {}
    results: list[dict] = []
    bar = tqdm(total=len(tasks), desc="Parallel benchmarks")

    def start_one(task):
        out_q: Queue = ctx.Queue(maxsize=1)
        p = ctx.Process(
            target=_worker_task,
            args=(task, out_q),
            daemon=False,
        )
        t0 = time.monotonic()
        p.start()
        running[p.pid] = (task, p, out_q, t0)  # type: ignore

    # launch initial batch
    while pending and len(running) < cores:
        start_one(pending.pop(0))

    try:
        while running:
            # pprint((len(pending), len(results), running))
            now = time.monotonic()

            # Collect finished processes
            finished_pids: list[int] = []
            for pid, (spec, p, out_q, t0) in running.items():
                if not p.is_alive():
                    bar.update(1)
                    try:
                        status, payload = out_q.get_nowait()
                    except Empty:
                        # could be os._exit, segfault, hard crash, etc.
                        res = _create_result(
                            Path(spec[0]).stem,
                            spec[1],
                            spec[2],
                            spec[3],
                            spec[4],
                            spec[5],
                            error="Process terminated unexpectedly",
                        )
                        results.append(res)
                        with open(output_file, "w") as f:
                            json.dump(results, f, indent=2)
                    else:
                        if status == "ok":
                            results.append(payload)
                            with open(output_file, "w") as f:
                                json.dump(results, f, indent=2)
                        elif status == "skip":
                            pass
                        elif status == "error":
                            res = _create_result(
                                Path(spec[0]).stem,
                                spec[1],
                                spec[2],
                                spec[3],
                                spec[4],
                                spec[5],
                                error=payload,
                            )
                            results.append(res)
                            with open(output_file, "w") as f:
                                json.dump(results, f, indent=2)

                    finished_pids.append(pid)

            for pid in finished_pids:
                running.pop(pid, None)

            # 2) Enforce timeouts
            timed_out_pids = []
            for pid, (task, p, out_q, t0) in list(running.items()):
                if p.is_alive() and (now - t0) > timeout:
                    bar.update(1)
                    bar.display(f"Timeout reached for task {task}")
                    p.terminate()
                    p.join(timeout=0.001)
                    if p.is_alive():
                        p.kill()
                        p.join(timeout=0.001)

                    res = _create_result(
                        Path(task[0]).stem,
                        task[1],
                        task[2],
                        task[3],
                        task[4],
                        task[5],
                        timeout=True,
                        check_time=now - t0,
                    )
                    results.append(res)
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=2)
                    timed_out_pids.append(pid)

            for pid in timed_out_pids:
                running.pop(pid, None)

            # 3) Start new ones if capacity
            while pending and len(running) < cores:
                start_one(pending.pop(0))

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, terminating workers...")
        for pid, (task, p, out_q, t0) in running.items():
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.01)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=0.01)
        raise

    return results


def _create_result(
    model_name,
    exact_mode,
    method,
    path_formula,
    query_type,
    threshold,
    timeout=False,
    error=None,
    value=None,
    marginal=None,
    states=None,
    transitions=None,
    check_time=None,
):
    """Create a result dict for any run"""
    success = error is None and not timeout
    result = {
        "model": model_name,
        "method": method,
        "arithmetic_mode": exact_mode,
        "query_type": query_type,
        "property": (
            f"Pmax=? [{path_formula}]"
            if query_type == "quantitative"
            else f"Pmax>={threshold} [{path_formula}]"
        ),
        "path_formula": path_formula,
        "success": success,
        "timeout": timeout,
        "error": error,
        "value": value,
        "marginal": marginal,
        "states": states,
        "transitions": transitions,
        "time": check_time,
    }
    if query_type == "bounded":
        result["threshold"] = threshold
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Storm model checking on BN models"
    )
    parser.add_argument(
        "--models", nargs="+", help="Specific models to benchmark (default: all models)"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bisection", "restart"],
        choices=["bisection", "bisection-advanced", "restart", "pi"],
        help="Conditional methods to test (default: bisection restart)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results (default: benchmark_results.json)",
        default="benchmark_results.json",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run benchmarks sequentially (default is parallel)",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=max(1, ((os.cpu_count() or 1) - 1)),
        help="Number of CPU cores to use for parallel execution",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each model checking call (default: 300)",
    )
    args = parser.parse_args()

    # Find all models
    bn_benchmark_dir = Path(__file__).parent.parent / "examples" / "BN-benchmarks-mdp"
    common_benchmark_dir = Path(__file__).parent.parent / "examples" / "transformed-mdp"
    mdp_benchmark_dir = Path(__file__).parent.parent / "examples" / "concrete-mdps"
    all_models = sorted(
        list(bn_benchmark_dir.glob("*.drn"))
        + list(common_benchmark_dir.glob("*.drn"))
        + list(mdp_benchmark_dir.glob("*.drn"))
    )

    # Filter models if specified
    if args.models:
        models = [m for m in all_models if m.stem in args.models]
        if not models:
            print(f"Error: No models found matching: {args.models}")
            return 1
    else:
        models = all_models

    # Sort models by file size (small to large)
    if args.sequential or args.cores >= 10:
        models = sorted(models, key=lambda m: m.stat().st_size)
    else:
        shuffle(models)

    print(f"Benchmarking {len(models)} models with methods: {args.methods}")
    print(f"Testing both exact, force-exact and float\n")

    # Build tasks - one task per single model checking call
    tasks = []
    for drn_file in models:
        model_name = drn_file.stem
        # Find properties for this model
        for key in PROPERTIES.keys():
            if model_name.startswith(key):
                model_key = key
                break
        else:
            print(f"Skipping model {model_name} - no properties defined.")
            continue
        for exact_mode in ["exact", "float", "force-exact"]:
            for method in args.methods:
                for path_formula in PROPERTIES[model_key]:
                    # Quantitative query
                    tasks.append(
                        (
                            str(drn_file),
                            exact_mode,
                            method,
                            path_formula,
                            "quantitative",
                            THRESHOLD,
                        )
                    )
                    # Bounded query
                    tasks.append(
                        (
                            str(drn_file),
                            exact_mode,
                            method,
                            path_formula,
                            "bounded",
                            THRESHOLD,
                        )
                    )

    # Collect all results
    all_results = []

    if args.sequential or len(tasks) == 0:
        # Run tasks sequentially
        for task in tqdm(list(tasks), desc="Sequential benchmarks"):
            res = _run_single_check(*task)
            if res:
                all_results.append(res)
                with open(args.output, "w") as f:
                    json.dump(all_results, f, indent=2)
    else:
        # Run tasks in parallel
        all_results = _run_in_parrallel(
            tasks, cores=args.cores, timeout=args.timeout, output_file=args.output
        )

    # Final save of results to JSON
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
