# %%
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np


def plot_distances(distances, samples, title, threshold=None, log=False, fname=None):
    plt.figure(figsize=(10, 6))
    plt.plot(samples, distances, marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    if threshold is not None:
        plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold")
    if log:
        plt.yscale("log")
    plt.grid(True)
    plt.show()


def plot_mult_distances(data_dict: dict, title, log=False):
    plt.figure(figsize=(10, 6))
    for key, data in sorted(data_dict.items(), reverse=True):
        line = plt.plot(data[1], data[0], marker="o", linestyle="-", label=key)
        if len(data[0]) >= 3 and key == "threshold":
            running_avg = [
                np.mean(data[0][max(0, i - (data[3] - 1)) : i + 1])
                for i in range(len(data[0]))
            ]
            plt.plot(data[1], running_avg, linestyle=":", color=line[0].get_color())

        if data[2] is not None:
            plt.axhline(y=data[2], color="r", linestyle="--", label="Threshold")

    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    if log:
        plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()


def main(stats_path="../../out/stats/2025-06-11_13-55-16"):
    path = Path(stats_path)
    stats_dicts: dict[tuple, dict] = {}
    if path.is_dir():
        paths = path.iterdir()
    else:
        paths = [path]

    for stat_path in paths:
        data = np.load(stat_path, allow_pickle=True).item()
        key = (
            data["args"]["mc"],
            (
                tuple(data["args"]["sys_vars"])
                if data["args"]["sys_vars"] is not None
                else None
            ),
            data["args"]["sam"],
            data["args"]["sim"],
            data["args"]["acas"],
            data["args"]["distance"],
        )

        if key not in stats_dicts:
            stats_dicts[key] = {}

        key2 = (
            data["args"]["stopping_criteria"]
            if "stopping_criteria" in data["args"]
            else "regression"
        )
        stats_dicts[key][key2] = data

    for key, exp_dict in stats_dicts.items():
        plot_data = {}
        for key2, data in exp_dict.items():
            if "distances" in data:  # Refinement
                distances = data["distances"]
                samples = data["samples_learned"]
            else:  # Regression
                distances = [data["target_dist"]]
                samples = [data["args"]["amount"]]

            cum_sam = np.cumsum(samples)
            if (
                "stopping_criteria" in data["args"]
                and data["args"]["stopping_criteria"] == "threshold"
            ):
                threshold = data["args"]["stopping_threshold"]
                patience = data["args"]["stopping_patience"]
            else:
                threshold = None
                patience = None

            plot_data[key2] = (distances, cum_sam, threshold, patience)

        plot_mult_distances(
            plot_data,
            f"{key[0]} with possible sys vars {key[1]} and distance {key[-1]}",
            log=True,
        )


main()
