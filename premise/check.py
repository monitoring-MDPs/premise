from argparse import ArgumentParser
from csv import DictReader
from fractions import Fraction
import math
from pathlib import Path
import re
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def check_step_state_equal(states: tuple[dict]) -> bool:
    first_state = states[0]
    for state in states[1:]:
        if any(first_state[k] != state[k] for k in first_state.keys() if k != "risk"):
            return False
    return True


def diff_traces_rational(
    traces: list[tuple[dict]],
) -> tuple[list[list[Fraction]], list[Fraction]]:
    diffs = []
    best_risks = []
    for step in zip(*traces):
        if not check_step_state_equal(step):
            raise ValueError(f"Traces have different states: {step}")

        risks = [s["risk"] for s in step]
        risk_counts = {r: risks.count(r) for r in set(risks)}
        correct_threshold = risks.count(True) >= risks.count(False)
        best_risk = max(risk_counts.items(), key=lambda x: x[1])[0]
        best_risks.append(best_risk)
        step_diffs = [
            (
                abs(r - best_risk)
                if isinstance(r, Fraction)
                else (r == correct_threshold) * 1000
            )
            for r in risks
        ]
        diffs.append(step_diffs)
    return diffs, best_risks


def diff_traces_floats(
    traces: list[tuple[dict]], correct_values: list[Fraction]
) -> list[list[float]]:
    diffs = []
    for step, correct in zip(zip(*traces), correct_values):
        if not check_step_state_equal(step):
            raise ValueError("Traces have different states")

        step_diffs = [abs(s["risk"] - correct) for s in step]
        diffs.append(step_diffs)
    return diffs


def compare_trace_files(paths: dict[str, Path]) -> dict[str, list[float | Fraction]]:
    real_indexes = {}
    double_indexes = {}
    real_traces = []
    double_traces = []
    for name, path in paths.items():
        with open(path, "r", newline="") as f:
            trace_reader = DictReader(f)
            trace = [row for row in trace_reader]
            if "." in trace[0]["risk"]:
                for step in trace:
                    step["risk"] = float(step["risk"])

                double_indexes[name] = len(double_traces)
                double_traces.append(trace)
            else:
                for step in trace:
                    if step["risk"] in ("True", "False"):
                        step["risk"] = step["risk"] == "True"
                    else:
                        step["risk"] = Fraction(step["risk"])

                real_indexes[name] = len(real_traces)
                real_traces.append(trace)

    real_diff, best_risks = diff_traces_rational(real_traces)
    double_diff = diff_traces_floats(double_traces, best_risks)
    diffs = {}
    for name, idx in real_indexes.items():
        diffs[name] = [d[idx] for d in real_diff]
    for name, idx in double_indexes.items():
        diffs[name] = [d[idx] for d in double_diff]
    return diffs


def compare_trace_folders(folders: list[Path]):
    seed_regex = r".*-(\d+)\.csv$"

    paths: dict[str, dict[str, Path]] = {}
    folder_names = []
    folder_speed_stats = {}
    for folder in folders:
        folder_name = folder.name
        folder_names.append(folder_name)
        stats_path = folder / "stats.out"
        if stats_path.exists():
            max_time = min_time = avg_time = None
            with open(stats_path, "r") as f:
                for line in f:
                    if line.startswith("max_time="):
                        max_time = float(line.split("=")[1].strip())
                    elif line.startswith("min_time="):
                        min_time = float(line.split("=")[1].strip())
                    elif line.startswith("avg_time="):
                        avg_time = float(line.split("=")[1].strip())
            folder_speed_stats[folder_name] = {
                "max_time": max_time,
                "min_time": min_time,
                "avg_time": avg_time,
            }

        for path in folder.glob("*.csv"):
            name = re.match(seed_regex, path.name)
            if not name:
                raise ValueError(f"Invalid trace file name: {path.name}")
            name = name.group(1)
            if name not in paths:
                paths[name] = {}
            paths[name][folder_name] = path

    folder_stats = {
        folder: {"avg_diff": [], "std_diff": [], "percent_wrong": [], "missing": 0}
        for folder in folder_names
    }

    for seed, seed_paths in tqdm(paths.items()):
        diffs = compare_trace_files(seed_paths)

        for name, diff in diffs.items():
            non_zero_diffs = [
                float(d)
                for d in diff
                if (isinstance(d, Fraction) and d > 0)
                or (isinstance(d, float) and not math.isclose(0.0, d))
            ]
            avg_diff = np.mean(non_zero_diffs) if non_zero_diffs else 0
            std_diff = np.std(non_zero_diffs, ddof=1) if len(non_zero_diffs) > 1 else 0
            percent_wrong = (len(non_zero_diffs) / len(diff)) * 100 if diff else 0

            folder_stats[name]["avg_diff"].append(avg_diff)
            folder_stats[name]["std_diff"].append(std_diff)
            folder_stats[name]["percent_wrong"].append(percent_wrong)

        for name in folder_names:
            if name not in diffs:
                folder_stats[name]["missing"] += 1

    for folder, stats in folder_stats.items():
        avg_diff = np.mean(stats["avg_diff"]) if stats["avg_diff"] else 0
        std_diff = np.mean(stats["std_diff"]) if stats["std_diff"] else 0
        median_percent_wrong = (
            np.median(stats["percent_wrong"]) if stats["percent_wrong"] else 0
        )

        print(f"Folder: {folder}")
        if folder in folder_speed_stats:
            speed_stats = folder_speed_stats[folder]
            print(
                f"  Time: [min={speed_stats['min_time']}, max={speed_stats['max_time']}, avg={speed_stats['avg_time']}]"
            )
        else:
            print("  Time: No stats available, method crashed before finishing.")
        print(f"  Average Difference: {avg_diff:.8f}")
        print(f"  Standard Deviation of Differences: {std_diff:.8f}")
        print(f"  Median Percentage Wrong: {median_percent_wrong:.2f}%")
        print(f"  Missing Traces: {stats['missing']}")

    # point with std line plot of timing of all methods
    labels = folder_names
    avg_times = [
        folder_speed_stats[folder]["avg_time"] if folder in folder_speed_stats else 0
        for folder in folder_names
    ]
    std_times = [
        (
            np.std(
                [
                    folder_speed_stats[folder]["min_time"],
                    folder_speed_stats[folder]["max_time"],
                ],
                ddof=1,
            )
            if folder in folder_speed_stats
            else 0.0
        )
        for folder in folder_names
    ]
    x = np.arange(len(labels))
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x,
        avg_times,
        yerr=std_times,
        fmt="o",
        ecolor="r",
        capsize=5,
        label="Average Time with Std Dev",
    )
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Time (s)")
    plt.title("Method Comparison: Average Time with Standard Deviation")
    plt.legend()
    plt.tight_layout()

    plt.savefig("method_comparison.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "folders",
        type=Path,
        nargs="+",
        help="Folders containing trace CSV files to compare",
    )
    args = parser.parse_args()
    compare_trace_folders(args.folders)
