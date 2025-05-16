# %%

from ast import mod
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
from premise.system import MCSystemUnderObservation
from premise.models import default_models
import numpy as np


def main_trace(
    stats_path="../../results/stats/2025-05-11_18-07-23/airportA-7-10-10-comp-reg-stats.npy",
):
    data = np.load(stats_path, allow_pickle=True).item()

    model_description = default_models[data["args"]["mc"]]
    suo = MCSystemUnderObservation(model_description, data["args"]["mc"])

    if "dist_traces" in data:
        target_all_dist = data["dist_traces"]
        for tl in target_all_dist[-1:]:
            sorted_traces = sorted(tl, key=lambda x: x[1][1])
            for t, (p, d, mr, tr) in sorted_traces[-1:]:
                print(suo.trace_to_str(t))
                print(f"Distance: {d}")
                print(f"Probability: {p}")
                print("Target risk:", tr)
                print("Monitored risk:", mr)

    else:
        target_all_dist = data["target_all_dist"]
        target_risks = data["target_risks"]
        regression_risks = data["regression_risks"]
        sorted_traces = sorted(target_all_dist, key=lambda x: x[1][1])
        for t, (p, d) in sorted_traces[-1:]:
            print(suo.trace_to_str(t))
            print(f"Distance: {d}")
            print(f"Probability: {p}")
            print("Target risk:", target_risks[t])
            print("Regression risk:", regression_risks[t])


main_trace("../../out/stats/2025-05-12_15-00-14/airportA-7-10-10_refinement.npy")
print("--------------------")
main_trace("../../out/stats/2025-05-12_15-00-14/airportA-7-10-10-comp-reg-stats.npy")

# %%
