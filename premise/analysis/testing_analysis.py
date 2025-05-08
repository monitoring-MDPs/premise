# %%
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
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
    plt.figure(figsize=(12, 12))
    plt.title("Receiver Operating Characteristic")
    for label, alarms in alarms_dict.items():
        if len(alarms) <= 0 or not any(alarms) or all(alarms):
            continue
        risks = risks_dict[label]
        fpr, tpr, threshold = metrics.roc_curve(alarms, risks)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=f"{label} [{np.sum(alarms)} / {len(alarms)} crashes] (AUC = {roc_auc:.2f})",
            color=color_to_rgb(label),
        )
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


def plot_risk_histogram(risks, alarms, fname=None):
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
    plt.title("Histogram of Model-based Risks")
    plt.legend()
    plt.grid(True)
    if fname:
        plt.savefig(fname)
    plt.show()


def main(
    stats_path="../../out/test-refine-HHH/carla-5-testing.npy", split_on_color=True
):
    data = np.load(stats_path, allow_pickle=True).item()
    risks = data["risks"]
    if "target_risks" not in data:
        target_risks = None
    else:
        target_risks = [float(r) for r in data["target_risks"]]
    alarms = data["alarms"]
    samples = data["samples"]

    print(f"Loaded {len(risks)} samples.")
    print(f"Crashes: {np.sum(alarms)} / {len(alarms)} ({100*np.mean(alarms):.2f}%)")
    print(
        f"Risks: min={np.min(risks):.4f}, max={np.max(risks):.4f}, mean={np.mean(risks):.4f}"
    )

    if split_on_color:
        colors = ["".join(c) for c in product(["L", "M", "H"], repeat=3)]

        c_alarms = {c: [] for c in colors}
        c_risks = {c: [] for c in colors}

        for i, t in enumerate(samples):
            c_alarms[t[0][0][2]].append(alarms[i])
            c_risks[t[0][0][2]].append(risks[i])

        # Print stats for each color
        for c in colors:
            if len(c_alarms[c]) == 0:
                continue
            print(
                f"{c}: Crashes: {np.sum(c_alarms[c])} / {len(c_alarms[c])} ({100*np.mean(c_alarms[c]):.2f}%)"
            )
            print(
                f"{c}: Risks: min={np.min(c_risks[c]):.4f}, max={np.max(c_risks[c]):.4f}, mean={np.mean(c_risks[c]):.4f}"
            )

        plot_mult_roc_curve(c_alarms, c_risks)

    plot_roc_curve(alarms, risks)
    plot_risk_histogram(risks, alarms)
    if target_risks is not None:
        plot_roc_curve(alarms, target_risks)
        plot_risk_histogram(target_risks, alarms)


if __name__ == "__main__":
    main()

# %%
