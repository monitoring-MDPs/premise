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
    line_styles = ['-', '--', ':']  
    style_index = 0
    #plt.title("Receiver Operating Characteristic")
    for label, alarms in alarms_dict.items():
        if len(alarms) <= 0 or not any(alarms) or all(alarms):
            continue
        risks = risks_dict[label]
        fpr, tpr, threshold = metrics.roc_curve(alarms, risks)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=f"{label}, (AUC = {roc_auc:.2f})",
            linewidth=6,
            linestyle=line_styles[style_index]
            #,color=color_to_rgb(label),
        )
        style_index = (style_index + 1) % len(line_styles)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.ylabel("True Positive Rate", fontsize=35)
    plt.xlabel("False Positive Rate", fontsize=35)
    plt.legend(loc="lower right", prop={'size': 22})
    plt.tick_params(axis='both', which='major', labelsize=20)
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


#def main(
#    stats_path="../../out/test-refine-HHH/carla-5-testing.npy", split_on_color=True
#):
def main_old(stats_path='/workspaces/premise/out/analysis/data.npy', split_on_color=False):
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

def main_alt(stats_path_1='/workspaces/premise/out/analysis/data.npy', stats_path_2='/workspaces/premise/out/analysis/data.npy', stats_path_3='/workspaces/premise/out/analysis/data.npy', split_on_color=False):
    data_1 = np.load(stats_path_1, allow_pickle=True).item()
    risks_1 = data_1["risks"]
    if "target_risks" not in data_1:
        target_risks = None
    else:
        target_risks_1 = [float(r) for r in data_1["target_risks"]]
    alarms_1 = data_1["alarms"]

    data_2 = np.load(stats_path_2, allow_pickle=True).item()
    risks_2 = data_2["risks"]
    if "target_risks" not in data_2:
        target_risks = None
    else:
        target_risks_2 = [float(r) for r in data_2["target_risks"]]
    alarms_2 = data_2["alarms"]

    data_3 = np.load(stats_path_3, allow_pickle=True).item()
    risks_3 = data_3["risks"]
    if "target_risks" not in data_3:
        target_risks = None
    else:
        target_risks_3 = [float(r) for r in data_3["target_risks"]]
    alarms_3 = data_3["alarms"]

    #print(f"Loaded {len(risks_1)} samples.")
    #print(f"Crashes: {np.sum(alarms_1)} / {len(alarms_1)} ({100*np.mean(alarms_1):.2f}%)")
    #print(
    #    f"Risks: min={np.min(risks_1):.4f}, max={np.max(risks_1):.4f}, mean={np.mean(risks_1):.4f}"
    #)


    plot_roc_curve(alarms_1, risks_1)
    plot_roc_curve(alarms_2, risks_2)
    plot_roc_curve(alarms_3, risks_3)

    #plot_risk_histogram(risks_1, alarms_1)

    if target_risks is not None:
        plot_roc_curve(alarms_1, target_risks_1)
        plot_roc_curve(alarms_2, target_risks_2)
        plot_roc_curve(alarms_3, target_risks_3)

        #plot_risk_histogram(target_risks_1, alarms_1)


def main(stats_path_1, stats_path_2, stats_path_3):
    alarms_dict = {}
    risks_dict = {}
    
    label_map = {
        1: "Model-based with refinement",
        2: "Model-based with uniform sampling",
        3: "Model-free"
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
        

if __name__ == "__main__":
    main('/workspaces/premise/premise/analysis/test_results/SnL-10x10_l15_ho5_ref.npy','/workspaces/premise/premise/analysis/test_results/SnL-10x10_l15_ho5_no_ref.npy','/workspaces/premise/premise/analysis/test_results/SnL-10x10_l15_ho5_reg.npy')

# %%
