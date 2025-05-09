# %%

import numpy as np


def plot_distances(distances, samples, title, threshold=None, log=False, fname=None):
    import matplotlib.pyplot as plt

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


def main(stats_path="../../out/others/ref-c.npy"):
    data = np.load(stats_path, allow_pickle=True).item()

    distances = data["distances"]
    samples = data["samples_iter"]
    cum_sam = np.cumsum(samples)
    if data["args"]["stopping_criteria"] == "threshold":
        threshold = data["args"]["stopping_threshold"]
    else:
        threshold = None

    plot_distances(distances, cum_sam, "Distance to Target Risk", threshold)
    plot_distances(
        distances,
        cum_sam,
        "Distance to Target Risk (log scale)",
        threshold,
        log=True,
    )
    print("Sample count:", data["sample_count"])


main()

# %%
