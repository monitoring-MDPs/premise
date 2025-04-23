# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


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


def plot_risk_histogram(risks, alarms, fname=None):
    plt.figure(figsize=(8, 6))
    plt.hist(
        [risks[i] for i in range(len(risks)) if alarms[i]],
        bins=30,
        alpha=0.5,
        label="Alarmed",
    )
    plt.hist(
        [risks[i] for i in range(len(risks)) if not alarms[i]],
        bins=30,
        alpha=0.5,
        label="Not Alarmed",
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
    stats_path="../../out/testing.npy",
):
    data = np.load(stats_path, allow_pickle=True).item()
    risks = data["risks"]
    alarms = data["alarms"]
    samples = data["samples"]

    print(f"Loaded {len(risks)} samples.")
    print(f"Alarms: {np.sum(alarms)} / {len(alarms)} ({100*np.mean(alarms):.2f}%)")
    print(
        f"Risks: min={np.min(risks):.4f}, max={np.max(risks):.4f}, mean={np.mean(risks):.4f}"
    )

    plot_roc_curve(alarms, risks)
    plot_risk_histogram(risks, alarms)


if __name__ == "__main__":
    main()
