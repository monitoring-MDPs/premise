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

    run_keys = set(k[0] for k in data_dict.keys())
    type_keys = set(k[1] for k in data_dict.keys())
    ls_map = {k: ["-", "-.", ":"][i % 4] for i, k in enumerate(type_keys)}
    col_map = {
        k: plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 10]
        for i, k in enumerate(run_keys)
    }

    for key, datas in sorted(data_dict.items(), reverse=True):
        distances_data = [data[0] for data in datas]
        transitions_data = [data[1] for data in datas]

        # Find common x range for interpolation
        min_x = max(min(transitions) for transitions in transitions_data)
        max_x = min(max(transitions) for transitions in transitions_data)
        x_values = np.linspace(min_x, max_x, 500)

        # Interpolate all runs to common x values
        interpolated_distances = []
        for distances, transitions in zip(distances_data, transitions_data):
            if log:
                distances = np.log10(distances)
            interpolated = np.interp(x_values, transitions, distances)
            if log:
                interpolated = np.power(10, interpolated)
            interpolated_distances.append(interpolated)

        # Calculate mean and std for interpolated y values
        distances_array = np.array(interpolated_distances)
        mean_distances = np.mean(distances_array, axis=0)
        std_distances = np.std(distances_array, axis=0)
        min_distances = np.min(distances_array, axis=0)
        max_distances = np.max(distances_array, axis=0)

        # Plot mean line
        plt.plot(
            x_values,
            mean_distances,
            linestyle=ls_map[key[1]],
            label=key,
            color=col_map[key[0]],
        )

        # Add shaded area for spread
        plt.fill_between(
            x_values,
            mean_distances - std_distances,
            mean_distances + std_distances,
            alpha=0.2,
            color=col_map[key[0]],
        )
        # plt.fill_between(
        #     x_values,
        #     min_distances,
        #     max_distances,
        #     alpha=0.2,
        #     color=col_map[key[0]],
        # )

        # Add threshold line if available
        if datas[0][2] is not None:
            plt.axhline(
                y=datas[0][2],
                color=col_map[key[0]],
                linestyle="--",
                c="red",
            )

    plt.title(title)
    plt.xlabel("Transitions")
    plt.ylabel("Distance")
    if log:
        plt.yscale("log")
    else:
        plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.show()


def main(
    stats_paths=[
        ("../../out/stats/2025-06-17_15-07-13", "obs start state, 10 prefixes"),
        ("../../out/stats/2025-06-18_09-54-18", "prefix start state, 10 prefixes"),
        ("../../out/stats/2025-06-18_15-46-39", "prefix start state, splitting"),
        ("../../out/stats/2025-06-23_11-19-27", "obs start state, splitting"),
    ]
):
    stats_dicts: dict[tuple, dict] = {}

    for stats_path, name in stats_paths:
        path = Path(stats_path)
        if path.is_dir():
            paths = path.iterdir()
        else:
            paths = [path]

        for stat_path in paths:
            data = np.load(stat_path, allow_pickle=True).item()
            model_key = (
                data["args"]["mc"],
                (
                    tuple(data["args"]["sys_vars"])
                    if data["args"]["sys_vars"] is not None
                    else None
                ),
                data["args"]["sam"],
                data["args"]["sim"],
                data["args"]["acas"],
                # data["args"]["distance"],
            )

            if model_key not in stats_dicts:
                stats_dicts[model_key] = {}

            learn_type_key = (
                name,
                (
                    data["args"]["stopping_criteria"]
                    if "stopping_criteria" in data["args"]
                    else "regression"
                ),
            )

            run_key = (data["args"]["run_id"] if "run_id" in data["args"] else 0,)

            if learn_type_key not in stats_dicts[model_key]:
                stats_dicts[model_key][learn_type_key] = {}

            stats_dicts[model_key][learn_type_key][run_key] = data

    for model_key, exp_dict in stats_dicts.items():
        plot_data: dict[tuple, list[tuple]] = {}
        for learn_type_key, runs in exp_dict.items():
            plot_data[learn_type_key] = []
            for run_key, data in runs.items():
                if "distances" in data:  # Refinement
                    distances = data["distances"]
                    samples = data["transitions_learned"]
                    sample_distances = [
                        [(x[1][1]) for x in dt] for dt in data["dist_traces"]
                    ]
                    sample_weights = [
                        [(x[1][0]) for x in dt] for dt in data["dist_traces"]
                    ]
                else:  # Regression
                    distances = [data["target_dist"]]
                    samples = [data["args"]["amount"] * data["args"]["length"]]
                    sample_distances = []
                    sample_weights = []

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

                plot_data[learn_type_key].append(
                    (
                        distances,
                        cum_sam,
                        threshold,
                        patience,
                        sample_distances,
                        sample_weights,
                    )
                )

        plot_mult_distances(
            plot_data,
            f"{model_key[0] or model_key[3]} with possible sys vars {model_key[1]}",
            log=True,
        )


main()
