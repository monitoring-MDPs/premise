# %%
import numpy as np
import matplotlib.pyplot as plt


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


def main(stats_path="../../out/tmp/airportA-7-10-10-comp-reg-stats.npy"):

    stats = data = np.load(stats_path, allow_pickle=True).item()

    target_risks = stats["target_risks"]
    regression_risks = stats["regression_risks"]
    sampled_risks = stats["sampled_risks"]
    weights = stats["weights"]

    traces = list(target_risks.keys())
    t = [target_risks[tr] for tr in traces]
    m = [regression_risks[tr] for tr in traces]
    s = [sampled_risks[tr] for tr in traces]
    w = [weights[tr] for tr in traces]

    # Scatter plots
    plot_scatter(m, t, "Regression Risk", "Target Risk", "Learned vs Target Risk")
    plot_scatter(m, s, "Regression Risk", "Sampling Risk", "Learned vs Sampling Risk")
    plot_scatter(t, s, "Target Risk", "Sampling Risk", "Target vs Sampling Risk")


if __name__ == "__main__":
    main()
