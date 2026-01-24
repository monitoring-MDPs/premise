#!/usr/bin/env python3
"""
Simple benchmark script for Storm model checking on BN-benchmarks-mdp models.
"""

import argparse
from fractions import Fraction
import json
import os
import re
import subprocess
from pathlib import Path
import sys
import time

BRP_PROPERTIES = [
    'F "success" || F "retries_MAX_min_1"',
    'F "success" || F "retries_MAX_min_3"',
    'F "target" || F "retries_MAX_min_1"',
    'F "target" || F "retries_MAX_min_3"',
]

CROWDS_PROPERTIES = [
    'F "observe0Greater1" || F "observeIGreater1"',
    'F "observeIGreater1" || F "observe0Greater1"',
]

WLAN_PROPERTIES = [
    'F "collision8" || F "collision2"',
    'F "collision1" || F "bothbackoff"'
]

COIN_PROPERTIES = [
    'F "finished" & "all_coins_equal_0"  || F "all_coins_equal_1"'
]

MONITORING_PROPERTY = [
    'F "_goal" || F "_end"',
]

# Define properties for each model
PROPERTIES = {
    "alarm": ['F "HYPOVOLEMIA1" || F "ANAPHYLAXIS1"', 'F "PAP0" || F "HRSAT1"'],
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
    "brp": BRP_PROPERTIES,
    "coin" : COIN_PROPERTIES,
    "crowds" : CROWDS_PROPERTIES,
    "wlan": WLAN_PROPERTIES,
    "airportA-7": MONITORING_PROPERTY,
    "airportB-7": MONITORING_PROPERTY,
    "evade-monitoring": MONITORING_PROPERTY,
    "hidden-incentive": MONITORING_PROPERTY,
    "refuel": MONITORING_PROPERTY,
    "patrol": MONITORING_PROPERTY,
}

# Threshold for bounded properties (can be changed)
THRESHOLD = 0.5

STORM_BINARY = "../storm/build/bin/storm"


def _build_tasks(args):

    # Find all models
    bn_benchmark_dir = Path(__file__).parent.parent / "examples" / "BN-benchmarks-mdp"
    common_benchmark_dir = Path(__file__).parent.parent / "examples" / "transformed-mdp"
    mdp_benchmark_dir = Path(__file__).parent.parent / "examples" / "concrete-mdps"
    monitoring_benchmark_dir = (
        Path(__file__).parent.parent / "examples" / "monitoring-cond-mdps"
    )
    all_models = sorted(
        #list(bn_benchmark_dir.glob("*.drn"))
        #+ list(common_benchmark_dir.glob("*.drn"))
        #+ list(mdp_benchmark_dir.glob("*.drn"))
        list(monitoring_benchmark_dir.glob("*.drn"))
    )

    # Filter models if specified
    if args.models:
        models = [m for m in all_models if m.stem in args.models]
        if not models:
            print(f"Error: No models found matching: {args.models}")
            return []
    else:
        models = all_models

    print(f"Benchmarking {len(models)} models with methods: {args.methods}")
    print(f"Testing both exact, exact-tolerance and float\n")

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
        for exact_mode in ["exact", "float", "exact-tolerance"]:
            for method in args.methods:
                if method == "restart" and exact_mode == "exact-tolerance":
                    continue
                for path_formula in PROPERTIES[model_key]:
                    # Quantitative query
                    print(f"SKIP QUANTITIVE: {args.skip_quantitative}")

                    if not args.skip_quantitative:
                        print(f"NOT SKIP QUANTITIVEs")
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

                    if exact_mode != "exact_tolerance" and method in ["bisection", "restart"]:
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

    return tasks


def _build_storm_command_from_task(
    drn_file,
    exact_mode,
    method,
    path_formula,
    query_type,
    threshold,
    timeout: int,
    index: int,
):
    """Build Storm command line from task parameters."""
    model_name = drn_file.stem

    # Build property string
    if query_type == "quantitative":
        prop_str = f"Pmax=? [{path_formula}]"
    else:  # bounded
        prop_str = f"Pmax>={threshold} [{path_formula}]"

    marginal_prop_str = f"Pmax=? [{path_formula.split('||')[1].strip()}]"

    # Base command
    cmd = [
        STORM_BINARY,
        "-drn",
        str(drn_file),
        "--prop",
        f"'{marginal_prop_str};{prop_str}'",
        "--timeout",
        str(timeout),
        "--dot-maxwidth",
        str(index)#,
        #"--debug",
    ]

    # Exact mode
    if exact_mode in ["exact", "exact-tolerance"]:
        cmd.append("--exact")

    if exact_mode == "exact":
        cmd.extend(["--minmax:precision", "0.0", "-condtol", "0"])
    else:
        cmd.extend(["--minmax:precision", "1e-6", "-condtol", "1e-6"])

    # Method
    if method == "bisection":
        cmd.extend(["--conditional", "bisection"])
    elif method == "bisection-advanced":
        cmd.extend(["--conditional", "bisection-advanced"])
    elif method == "bisection-pt":
        cmd.extend(["--conditional", "bisection-pt"])
    elif method == "bisection-advanced-pt":
        cmd.extend(["--conditional", "bisection-advanced-pt"])
    elif method == "restart":
        cmd.extend(["--conditional", "restart"])
    elif method == "pi":
        cmd.extend(["--conditional", "pi"])

    return cmd


def _write_commands_file(tasks, timeout, commands_file_path):
    """Build a commands file for Storm from the list of tasks."""
    with open(commands_file_path, "w") as f:
        for i, task in enumerate(tasks):
            drn_file, exact_mode, method, path_formula, query_type, threshold = task
            drn_path = Path(drn_file)
            cmd = _build_storm_command_from_task(
                drn_path,
                exact_mode,
                method,
                path_formula,
                query_type,
                threshold,
                timeout,
                i,
            )
            f.write(" ".join(cmd) + "\n")


def _run_parrallel_on_commands_file(commands_file_path, cores):
    """run GNU parallel on the commands file in the background while connecting stdout and stderr. Returing the process."""
    # parallel --verbose --results out/$now/logs/{#} --ungroup --eta --no-run-if-empty --joblog out/$now/joblog.tsv --jobs cores < commands_file
    cmd = [
        "parallel",
        "--verbose",
        "--ungroup",
        "--eta",
        "--no-run-if-empty",
        "--jobs",
        str(cores),
        "--joblog",
        str(commands_file_path.parent / "joblog.tsv"),
        "--results",
        str(commands_file_path.parent / "logs") + "/{#}/",
        "<",
        str(commands_file_path),
    ]
    process = subprocess.Popen(
        " ".join(cmd),
        shell=True,
        text=True,
    )
    return process


def _parse_storm_value(output: str):
    result_match = re.search(r"Result \(for initial states\): ([^ \n]+)", output)

    if result_match:
        value_str: str = result_match.group(1)
        # Handle rational numbers (fraction format)
        if value_str == "true":
            value = 1.0
        elif value_str == "false":
            value = 0.0
        else:
            value = float(Fraction(value_str))

        return value
    elif "Result (for initial states):" in output:
        print(f"Could not parse result value: {output}")
        return None
    else:
        return None


def _parse_storm_output(output: str, tasks: list, parrallel_index: int):
    """
    Parse Storm output and create result dictionary.

    Args:
        output: Storm's stdout/stderr output
        task: Tuple of (drn_file, exact_mode, method, path_formula, query_type, threshold)

    Returns:
        Dictionary with result information
    """
    sys.set_int_max_str_digits(100000)

    task_index = re.search(r"--dot-maxwidth (\d+)", output)
    if task_index:
        task_index = int(task_index.group(1))
    else:
        print("Could not find task index in output.")
        return {}

    task = tasks[task_index]

    drn_file, exact_mode, method, path_formula, query_type, threshold = task
    model_name = Path(drn_file).stem

    # Initialize result dictionary with task information
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
        "success": False,
        "timeout": False,
        "unfinished": False,
        "error": None,
        "value": None,
        "marginal": None,
        "states": None,
        "transitions": None,
        "time": None,
        "iterations": None,
        "normalform_states": None,
        "index": parrallel_index,
        "task_index": task_index,
    }

    if query_type == "bounded":
        result["threshold"] = threshold

    # Check if output is empty or incomplete (process not yet done)
    if not output or not output.strip():
        result["error"] = "No output (process may not be done yet)"
        result["unfinished"] = True
        return result

    # Check for timeout indicators
    if "The program received signal 14" in output or "Result till abort" in output:
        result["timeout"] = True

        # Extract time for model checking
        time_match = re.search(r"Time for model checking: ([\d.]+)s", output)
        if time_match:
            result["time"] = float(time_match.group(1))

    # Check for successful result
    elif "Result (for initial states):" in output:
        result["success"] = True

        outputs = output.split("Model checking property")[1:]

        if len(outputs) != 2:
            result["error"] = "Could not find both marginal and main results in output"
        else:
            # Extract marginal result
            marginal_output = outputs[0]
            marginal_value = _parse_storm_value(marginal_output)
            if marginal_value is not None:
                result["marginal"] = marginal_value
            else:
                result["error"] = "Could not parse marginal result value"
                result["unfinished"] = True
                result["success"] = False

            # Extract main result
            main_output = outputs[1]
            main_value = _parse_storm_value(main_output)
            if main_value is not None:
                result["value"] = main_value
            else:
                result["error"] = "Could not parse main result value"
                result["unfinished"] = True
                result["success"] = False

            # Extract time for model checking
            time_match = re.search(r"Time for model checking: ([\d.]+)s", outputs[1])
            if time_match:
                result["time"] = float(time_match.group(1))
            else:
                result["error"] = "Could not find model checking time in output"

    # Extract number of iterations in bisection "Bisection method converged after 159 iterations."
    iterations_match = re.search(
        r"Bisection method converged after (\d+) iterations", output
    )
    if iterations_match:
        result["iterations"] = int(iterations_match.group(1))

    # Extract model statistics
    states_match = re.search(r"States:\s+(\d+)", output)
    if states_match:
        result["states"] = int(states_match.group(1))

    # Extract normal form states "Analyzing normal form with 1208 maybe states"
    normalform_states_match = re.search(
        r"Analyzing normal form with (\d+) maybe states", output
    )
    if normalform_states_match:
        result["normalform_states"] = int(normalform_states_match.group(1))

    # Extract transitions
    transitions_match = re.search(r"Transitions:\s+(\d+)", output)
    if transitions_match:
        result["transitions"] = int(transitions_match.group(1))

    # Check for other errors
    if not result["success"] and not result["timeout"]:
        if "ERROR" in output:
            # Extract error message
            error_lines = [line for line in output.split("\n") if "ERROR" in line]
            if error_lines:
                result["error"] = error_lines[0]
        else:
            # Process started model checking but didn't finish
            result["error"] = "Model checking incomplete (process may not be done yet)"
            result["unfinished"] = True

    return result


def _write_results(tasks, output_dir, check_all=False):
    results_json_path = output_dir / "results.json"

    all_results = []

    for i in range(len(tasks)):
        stdout = output_dir / "logs" / str(i + 1) / "stdout"
        stderr = output_dir / "logs" / str(i + 1) / "stderr"
        if stdout.exists() and stderr.exists():
            with open(stdout, "r") as f:
                output_stdout = f.read()
            with open(stderr, "r") as f:
                output_stderr = f.read()
            output = output_stdout + "\n" + output_stderr
            result = _parse_storm_output(output, tasks, i + 1)
            all_results.append(result)
            with open(results_json_path, "w") as f:
                json.dump(all_results, f, indent=4)
        elif check_all:
            print(f"Log file {stdout} does not exist for task {i}.")


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
        default=["bisection", "bisection-pt", "restart", "bisection-advanced", "bisection-advanced-pt" ],
        choices=["bisection", "bisection-advanced", "bisection-pt", "bisection-advanced-pt", "restart", "pi"],
        help="Conditional methods to test (default: bisection restart)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output folder",
        default=f"out/{time.strftime('%Y-%m-%d-%H-%M-%S')}_bn_benchmark/",
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
    parser.add_argument(
        "--reparse",
        action="store_true",
        help="Only reparse existing log files without rerunning benchmarks",
    )
    parser.add_argument(
        "--skip-quantitative",
        action="store_true",
        help="Skip quantitative tests"
    )

    args = parser.parse_args()


    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reparse:
        print("Reparsing existing log files...")
        tasks = _build_tasks(args)
        _write_results(tasks, output_dir, check_all=False)
        return

    tasks = _build_tasks(args)
    _write_commands_file(
        tasks,
        args.timeout,
        output_dir / "storm_commands.txt",
    )
    process = _run_parrallel_on_commands_file(
        output_dir / "storm_commands.txt",
        args.cores,
    )

    try:
        while process.poll() is None:
            _write_results(tasks, output_dir)
            time.sleep(20)
    except KeyboardInterrupt:
        print("Benchmarking interrupted by user. Terminating processes...")
        process.terminate()
        process.wait()

    _write_results(tasks, output_dir, check_all=True)


if __name__ == "__main__":
    main()
