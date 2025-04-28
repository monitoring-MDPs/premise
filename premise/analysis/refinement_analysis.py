# %%

import numpy as np


def plot_distances(distances, title, log=False, fname=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(distances)), distances, marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    if log:
        plt.yscale("log")
    plt.grid(True)
    plt.show()


def main(stats_path="../../out/refinement.npy"):
    data = np.load(stats_path, allow_pickle=True).item()

    distances = data["distances"]
    plot_distances(
        distances,
        "Distance to Target Risk",
    )
    plot_distances(
        distances,
        "Distance to Target Risk (log scale)",
        log=True,
    )
    print("Sample count:", data["sample_count"])


main()

# %%
