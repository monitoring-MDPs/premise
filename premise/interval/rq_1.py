import argparse
from matplotlib import pyplot as plt
import numpy as np
from sympy import Line2D
from tqdm import tqdm, trange
import os
import matplotlib.ticker as ticker


import numpy as np
import argparse

import argparse

import numpy as np
from tqdm import tqdm
import os


from premise.interval.loading import (
    build_suo,
    build_suo_args_parser,
    load_imc,
)
from premise.interval.conformence import test_monitor
from premise.interval.interval import (
    Trace,
    create_monitor,
)


def stats_true(horizon, initial_amount, testing_samples, suo):

    target_risks = []

    for trace in tqdm(testing_samples):
        sub_trace: Trace = trace[:initial_amount]

        target_risk, risks = test_monitor(
            suo.create_target_monitor(),
            [sub_trace],
            with_tqdm=False,
            intermediate_results=True,
        )

        print(
            f"Target risk for trace {sub_trace}: {target_risk[sub_trace]} with intermediate results {risks}"
        )

        target_risks.append(float(target_risk[sub_trace]))

    return target_risks


def aggregated_stats_imc(
    path, stats_path, initial_amount, horizon, args, testing_samples
):

    imc_risks = {}

    imc_transition_counts = {}

    for x in range(1, 11):

        print(f"Experiment number {x}")
        if not os.path.exists(f"{stats_path}-{x}.npy"):
            print(f"Statistics file for experiment {x} does not exist.")
            continue
        statistics = np.load(f"{stats_path}-{x}.npy", allow_pickle=True)

        obj = statistics.item()
        imc_transition_count = obj["transitions_learned"]

        sum = 0
        total_imc_transition_count = []
        for z in range(len(imc_transition_count)):
            sum += imc_transition_count[z]
            total_imc_transition_count.append(sum)

        imc_transition_counts[str(x)] = total_imc_transition_count

        for y in trange(1, len(imc_transition_count) + 1):
            initial_distribution = f"{path}-{x}-{y}-initial_interval.npy"
            transition_intervals = f"{path}-{x}-{y}-interval.npy"

            args.trans_path = transition_intervals
            args.init_path = initial_distribution
            args.dump = None
            args.exact = True
            args.precision = 1e-6

            transition_intervals, initial_distribution = load_imc(args)

            # Build the premise monitor on the learned model
            mon, mon_comps = create_monitor(
                transition_intervals,
                initial_distribution,
                "min",
                True,
                horizon,
                args.dump,
                args.verbose,
                use_exact=args.exact,
                precision=args.precision,
            )

            imc_risks[f"{x}-{y}"] = []

            for t in testing_samples:
                sub_trace: Trace = t[:initial_amount]
                risk = test_monitor(
                    mon,
                    [sub_trace],
                    obs_func=lambda x: mon_comps.observation_map[x],
                    skip_initial=True,
                    with_tqdm=False,
                )[sub_trace]

                imc_risks[f"{x}-{y}"].append(float(risk))

    return imc_risks, imc_transition_counts


def distance_graph(
    target_risks, imc_risks, imc_transition_counts, testing_samples_weights
):

    distance_stats = {}

    for key in imc_risks.keys():
        total_distance = 0
        for x in range(len(target_risks)):
            total_distance += testing_samples_weights[x] * abs(
                imc_risks[key][x] - target_risks[x]
            )
            distance_stats[key] = total_distance

    distance_graph_data = {}
    for x in range(1, 11):
        distance_graph_data[x] = []

    for key in distance_stats.keys():
        for x in range(1, 11):
            if key.split("-")[0] == str(x):
                distance_graph_data[x].append(distance_stats[key])

    graph_data = []
    for x in range(1, 11):
        graph_data.append([distance_graph_data[x], imc_transition_counts[str(x)]])

    log = False
    plt.figure()
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(graph_data[0][1], graph_data[0][0])
    ax.plot(graph_data[1][1], graph_data[1][0])
    ax.plot(graph_data[2][1], graph_data[2][0])
    ax.plot(graph_data[3][1], graph_data[3][0])
    ax.plot(graph_data[4][1], graph_data[4][0])
    ax.plot(graph_data[5][1], graph_data[5][0])
    ax.plot(graph_data[6][1], graph_data[6][0])
    ax.plot(graph_data[7][1], graph_data[7][0])
    ax.plot(graph_data[8][1], graph_data[8][0])
    ax.plot(graph_data[9][1], graph_data[9][0])

    # IMC AVERAGE PERFORMANCE
    transitions_data = []
    distance_data = []

    for entry in graph_data:
        transitions = entry[1]
        distance_daum = entry[0]

        transitions_data.append(transitions)
        distance_data.append(distance_daum)

    # Find common x range for interpolation
    min_x = max(min(transitions) for transitions in transitions_data)
    max_x = min(max(transitions) for transitions in transitions_data)
    x_values = np.linspace(min_x, max_x, 500)

    # Interpolate all runs to common x values
    interpolated_auc = []
    for auc, transitions in zip(distance_data, transitions_data):
        if log:
            auc = np.log10(auc)
        interpolated = np.interp(x_values, transitions, auc)
        if log:
            interpolated = np.power(10, interpolated)
        interpolated_auc.append(interpolated)

    # Calculate mean and std for interpolated y values
    distance_array = np.array(interpolated_auc)
    mean_distance = np.mean(distance_array, axis=0)
    std_distance = np.std(distance_array, axis=0)

    # Plot mean line
    ax.plot(
        x_values,
        mean_distance,
        color="blue",
        label="IMC",
        linewidth=5,
        linestyle=":",
    )

    # Add shaded area for spread
    ax.fill_between(
        x_values,
        mean_distance - std_distance,
        mean_distance + std_distance,
        alpha=0.2,
        color="blue",
    )

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((4, 4))  # Force 10^4 scale
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="both", labelsize=20)
    ax.xaxis.get_offset_text().set_size(20)

    ax.set_xlabel("State count", fontsize=30)
    ax.set_ylabel("Distance to Target", fontsize=30)
    ax.legend(loc="lower right")
    if log:
        plt.yscale("log")
    else:
        plt.ylim(bottom=0)
    ax.grid(True)
    plt.subplots_adjust(bottom=0.25)
    plt.title(f"{args.mc}", fontsize=30)

    plt.savefig(
        "/workspaces/premise/premise/analysis/rq_1_distance_to_RRF.pdf", dpi=300
    )
    plt.show()


def overestimation_graph(target_risks, imc_risks, imc_transition_counts):

    plt.figure()
    plt.plot([0, 1], [0, 1], "r--")

    imc_ys = []

    for key in imc_risks.keys():
        imc_ys.append(int(key.split("-")[1]))

    for key in imc_risks.keys():
        if key.split("-")[1] == max(imc_ys):
            print(key)
            plt.scatter(
                imc_risks[key], target_risks, color="blue", marker="o", label="IMC"
            )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="blue",
            label="IMC",
            markerfacecolor="blue",
            markersize=8,
        ),
    ]
    plt.legend(handles=legend_elements)

    plt.savefig("/workspaces/premise/premise/analysis/rq_1_overestimation.pdf", dpi=300)
    plt.show()


def main_imc(args: argparse.Namespace):

    suo, initial_amount, horizon = build_suo(args)

    length = initial_amount + horizon

    testing_samples = []
    testing_samples_weights = []

    for x in range(args.testing_samples):
        path = suo.generate_random_traces_with_prob([], length)
        testing_samples.append(tuple(path[0][0]))
        testing_samples_weights.append(float(path[0][1]))

    target_risks = stats_true(horizon, initial_amount, testing_samples, suo)
    imc_risks, imc_transition_counts = aggregated_stats_imc(
        args.imc_model, args.imc_stats, initial_amount, horizon, args, testing_samples
    )
    distance_graph(
        target_risks, imc_risks, imc_transition_counts, testing_samples_weights
    )
    overestimation_graph(target_risks, imc_risks, imc_transition_counts)


def build_learning_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Learning Parameters")

    group.add_argument(
        "--model_name",
        type=str,
        default="MC",
        help="Name of the model (first letters code).",
    )
    group.add_argument(
        "-s",
        "--testing_samples",
        type=int,
        default=100,
        help="Total number of samples used in learning",
    )


def testing_argsparser():
    parser = argparse.ArgumentParser(description="Learn an IMC")
    build_suo_args_parser(parser)
    build_learning_parser(parser)

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (can be used multiple times)",
    )

    parser.add_argument("--imc-model", type=str, help="Path imc models")

    parser.add_argument("--imc-stats", type=str, help="Path imc stats")
    parser.add_argument("--mc-model", type=str, help="Path mc models")
    parser.add_argument("--mc-stats", type=str, help="Path mc stats")

    return parser


if __name__ == "__main__":
    parser = testing_argsparser()
    args = parser.parse_args()
    main_imc(args)


# python -m premise.interval.rq_1 --mc SnL-10x10 --imc-model /workspaces/premise/out/models/2025-07-10_07-55-18/SnL-10x10-comp-no-ref --imc-stats /workspaces/premise/out/stats/2025-07-10_07-55-18/SnL-10x10-comp-noref-stats
