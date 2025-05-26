# %%
from itertools import combinations
from premise.models import default_models
from premise.system import MCSystemUnderObservation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import metrics


def color_to_rgb(color):
    c_map = {
        "L": 0,
        "M": 0.5,
        "H": 0.8,
    }

    return (c_map[color[0]], c_map[color[1]], c_map[color[2]])


def plot_roc_curve(alarms, risks, fname=None):
    fpr, tpr, threshold = metrics.roc_curve(alarms, risks)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    if fname:
        plt.savefig(fname)
    plt.show()
    print(f"AUC: {roc_auc:.4f}")


def plot_mult_roc_curve(alarms_dict, risks_dict, fname=None):
    plt.figure(figsize=(8, 6))
    line_styles = ["-"]
    style_index = 0
    # plt.title("Receiver Operating Characteristic")
    for label, alarms in alarms_dict.items():
        if len(alarms) <= 0 or not any(alarms) or all(alarms):
            continue
        risks = risks_dict[label]
        fpr, tpr, thresholds = metrics.roc_curve(alarms, risks)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=f"{label}, (AUC = {roc_auc:.2f})",
            # linewidth=6,
            linestyle=line_styles[style_index],
            # ,color=color_to_rgb(label),
        )
        style_index = (style_index + 1) % len(line_styles)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right")
    plt.tick_params(axis="both")
    plt.grid(True)
    if fname:
        plt.savefig(fname)
    plt.show()


def plot_risk_histogram(risks, alarms, title, fname=None):

    plt.figure(figsize=(8, 6))
    bins = np.linspace(0, 1, 100)
    plt.hist(
        [risks[i] for i in range(len(risks)) if alarms[i]],
        bins=bins,  # type: ignore
        alpha=0.5,
        label="Crash",
    )
    plt.hist(
        [risks[i] for i in range(len(risks)) if not alarms[i]],
        bins=bins,  # type: ignore
        alpha=0.5,
        label="Not Crashed",
    )
    plt.xlabel("Risk")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if fname:
        plt.savefig(fname)
    plt.show()


def inspect_traces(traces, alarms, risks_dict, suo, sample_length):
    print("Alarmed traces:")
    for i, trace in enumerate(traces):
        if alarms[i]:
            print(f"Trace {i}: {suo.trace_to_str(trace[: sample_length])}")
            print(f"Horizon: {suo.trace_to_str(trace[sample_length:])}")
            for k in risks_dict.keys():
                print(f"{k} risk: {risks_dict[k][i]}")

    print("Not alarmed traces:")
    for i, trace in enumerate(traces):
        if not alarms[i]:
            print(f"Trace {i}: {suo.trace_to_str(trace[: sample_length])}")
            print(f"Horizon: {suo.trace_to_str(trace[sample_length:])}")
            for k in risks_dict.keys():
                print(f"{k} risk: {risks_dict[k][i]}")


def main(stats_path="../../out/tmp/testing-a-dtmc-umse.npy"):
    matplotlib.rcParams["figure.dpi"] = 300

    data = np.load(stats_path, allow_pickle=True)

    if "args" not in data:
        data["args"] = {"mc": "airportA-7-10-10"}
    model_description = default_models[data["args"]["mc"]]
    suo = MCSystemUnderObservation(model_description, data["args"]["mc"])

    samples = data["samples"]
    alarms = data["alarms"]

    risks_dict = {}
    alarm_dict = {}
    for name, risks in data["risks"].items():
        risks_dict[name] = risks
        alarm_dict[name] = alarms

    print(f"Loaded {len(samples)} samples.")
    print(f"Crashes: {np.sum(alarms)} / {len(alarms)} ({100*np.mean(alarms):.2f}%)")

    plot_mult_roc_curve(alarm_dict, risks_dict)
    for k in alarm_dict.keys():
        plot_risk_histogram(risks_dict[k], alarm_dict[k], f"Histogram of {k} risks")

    # Plot all scatter plots in a grid dynamic on the number of keys
    plots = [(k1, k2) for k1, k2 in combinations(risks_dict.keys(), 2)]
    num_cols = 3
    num_rows = int(np.ceil(len(plots) / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    fig.subplots_adjust(hspace=0.5)
    for i, (key1, key2) in enumerate(plots):
        ax = axs[i // num_cols, i % num_cols]
        ax.scatter(
            risks_dict[key1], risks_dict[key2], c=alarms, cmap="coolwarm", alpha=0.5
        )
        ax.set_xlabel(key1)
        ax.set_ylabel(key2)
        ax.set_title(f"Scatter plot of {key1} vs {key2}")
        ax.plot([0, 1], [0, 1], "r--")
        correlation = np.corrcoef(risks_dict[key1], risks_dict[key2])[0, 1]
        ax.annotate(
            f"Corr: {correlation:.2f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
        )
        ax.grid(True)
    plt.tight_layout()
    plt.show()

    # weird_indices = []
    # for i, _ in enumerate(samples):
    #     if risks_dict["target"][i] > 0.9:
    #         weird_indices.append(i)
    #
    # risks_dict["target"] = data["target_risks"]
    #
    # inspect_traces(
    #     [samples[i] for i in weird_indices],
    #     [alarms[i] for i in weird_indices],
    #     {k: [rs[i] for i in weird_indices] for k, rs in risks_dict.items()},
    #     suo,
    #     data["args"]["sample_length"] if "sample_length" in data["args"] else 25,
    # )


if __name__ == "__main__":
    main()

# %%


def main_paper(stats_path_1, stats_path_2, stats_path_3):
    alarms_dict = {}
    risks_dict = {}

    label_map = {
        1: "Model-based with refinement",
        2: "Model-based with uniform sampling",
        3: "Model-free",
    }

    paths = [stats_path_1, stats_path_2, stats_path_3]

    for idx, path in enumerate(paths, start=1):
        label = label_map[idx]
        data = np.load(path, allow_pickle=True)

        # If it's a numpy array that wraps a dict (common with np.save on dict), unpack it
        if not isinstance(data, dict):
            data = data.item()

        alarms = data["alarms"]
        risks = data["risks"]

        alarms_dict[label] = alarms
        risks_dict[label] = risks

        if "target_risks" in data:
            target_label = f"{label} (Target)"
            alarms_dict[target_label] = alarms
            risks_dict[target_label] = [float(r) for r in data["target_risks"]]

    plot_mult_roc_curve(alarms_dict, risks_dict)

    for k in alarms_dict.keys():
        plot_risk_histogram(risks_dict[k], alarms_dict[k])
