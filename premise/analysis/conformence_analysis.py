# %%
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_stats(stats_path):
    stats = np.load(stats_path, allow_pickle=True).item()
    return stats


def extract_risks(stats):
    target_risks = stats["target_risks"]
    monitored_risks = stats["monitored_risks"]
    sampled_risks = stats["sampled_risks"]
    weights = stats.get("weights", None)
    return target_risks, monitored_risks, sampled_risks, weights


def risks_to_lists(target_risks, monitored_risks, sampled_risks):
    traces = list(target_risks.keys())
    t = [target_risks[tr] for tr in traces]
    m = [monitored_risks[tr] for tr in traces]
    s = [sampled_risks[tr] for tr in traces]
    return t, m, s


def plot_scatter(x, y, xlabel, ylabel, title, fname=None):
    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    # Scatter plot
    plt.scatter(x, y, alpha=0.6)
    # Save current axis limits
    xlim = plt.xlim()
    ylim = plt.ylim()
    # Plot diagonal line without affecting limits
    plt.plot([0, 1], [0, 1], "r--", linewidth=1, label="y=x")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if fname:
        plt.savefig(fname)
    plt.show()


def main(stats_path="../../out/conformence.npy"):

    stats = load_stats(stats_path)
    target_risks, monitored_risks, sampled_risks, weights = extract_risks(stats)
    t, m, s = risks_to_lists(target_risks, monitored_risks, sampled_risks)

    # Scatter plots
    plot_scatter(m, t, "Monitor Risk", "Target Risk", "Learned vs Target Monitor Risk")
    plot_scatter(m, s, "Monitor Risk", "Sampling Risk", "Learned vs Sampling Risk")
    plot_scatter(
        t,
        s,
        "Target Risk",
        "Sampling Risk",
        "Target vs Sampling Risk",
    )

    # Compare distances
    print("Distance to sampling:", stats.get("sample_dist"))
    print("Distance to target:", stats.get("target_dist"))

    # Optionally, plot histogram of all distances if available
    if "target_all_dist" in stats and "sample_all_dist" in stats:
        plt.figure()
        _, bins, _ = plt.hist(
            [d[1][1] for d in stats["sample_all_dist"]],
            bins=30,
            alpha=0.5,
            label="Sampling",
        )
        plt.hist(
            [d[1][1] for d in stats["target_all_dist"]],
            bins,  # type: ignore
            alpha=0.5,
            label="Target",
        )
        plt.xlabel("Distance")
        plt.ylabel("Count")
        plt.title("Distribution of Trace Distances")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()

# %%
