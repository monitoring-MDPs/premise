import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from tqdm import tqdm, trange
import matplotlib.ticker as ticker
import pickle
import numpy as np
from premise.models import *
from premise.interval.maximum_likelihood import *
from itertools import groupby
from operator import itemgetter


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
from premise.interval.maximum_likelihood import dict_to_pomdp, create_mle_monitor


def aggregted_alarms(testing_samples):
    alarms = []

    for trace in tqdm(testing_samples):
        alarms.append(any([s[2] for s in trace]))

    alarms = np.array(alarms).astype(int)

    return alarms


def stats_true(horizon, initial_amount, testing_samples, suo):

    target_risks = []

    mon = suo.create_target_monitor()

    for trace in tqdm(testing_samples):
        sub_trace: Trace = trace[:initial_amount]

        target_risk, risks = test_monitor(
            mon,
            [sub_trace],
            with_tqdm=False,
            intermediate_results=True,
        )

        target_risks.append(float(target_risk[sub_trace]))

    return target_risks


def aggregated_stats_imc(
    method, coarse, stats_path, initial_amount, horizon, args, testing_samples
):
    imc_risks = {}

    imc_transition_counts = {}

    for x in range(1, 11):
        print(f"Experiment number {x}")
        try:
            if coarse:
                """if high_st:
                    if method == "noref":
                        if args.mc == " SnLw-10x10":
                            statistics = np.load(
                                f"{stats_path}/high-st-SnL-coarse_norefinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "evadeV-6-3-coarse":
                            statistics = np.load(
                                f"{stats_path}/high-st-evadeV-6-3-coarse_norefinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        else:
                            statistics = np.load(
                                f"{stats_path}/high-st-{args.mc}-coarse_norefinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                    if method == "ref":
                        if args.mc == " SnLw-10x10":
                            statistics = np.load(
                                f"{stats_path}/high-st-SnL-coarse_refinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "evadeV-6-3-coarse":
                            statistics = np.load(
                                f"{stats_path}/high-st-evadeV-6-3-coarse_refinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        else:
                            statistics = np.load(
                                f"{stats_path}/high-st-{args.mc}-coarse_refinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                    if method == "refsplit":
                        if args.mc == " SnLw-10x10":
                            statistics = np.load(
                                f"{stats_path}/high-st-SnL-coarse_refsplitinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "evadeV-6-3-coarse":
                            statistics = np.load(
                                f"{stats_path}/high-st-evadeV-6-3-coarse_refsplitinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        else:
                            statistics = np.load(
                                f"{stats_path}/high-st-{args.mc}-coarse_refsplitinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                else:"""
                if method == "noref":
                    if args.mc == " SnLw-10x10":
                        statistics = np.load(
                            f"{stats_path}/SnL-coarse_norefinement-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    elif args.mc == "evadeV-6-3-coarse":
                        statistics = np.load(
                            f"{stats_path}/evadeV-6-3-coarse_norefinement-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    else:
                        statistics = np.load(
                            f"{stats_path}/{args.mc}-coarse_norefinement-stats-{x}.npy",
                            allow_pickle=True,
                        )
                if method == "ref":
                    if args.mc == " SnLw-10x10":
                        statistics = np.load(
                            f"{stats_path}/SnL-coarse_refinement-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    elif args.mc == "evadeV-6-3-coarse":
                        statistics = np.load(
                            f"{stats_path}/evadeV-6-3-coarse_refinement-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    else:
                        statistics = np.load(
                            f"{stats_path}/{args.mc}-coarse_refinement-stats-{x}.npy",
                            allow_pickle=True,
                        )
                if method == "refsplit":
                    if args.mc == " SnLw-10x10":
                        statistics = np.load(
                            f"{stats_path}/SnL-coarse_refsplitinement-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    elif args.mc == "evadeV-6-3-coarse":
                        statistics = np.load(
                            f"{stats_path}/evadeV-6-3-coarse_refsplitinement-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    else:
                        statistics = np.load(
                            f"{stats_path}/{args.mc}-coarse_refsplitinement-stats-{x}.npy",
                            allow_pickle=True,
                        )
            else:
                """if high_st:
                    if method == "noref":
                        statistics = np.load(
                            f"{stats_path}/high-st-{args.mc}-comp-noref-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    if method == "ref":
                        statistics = np.load(
                            f"{stats_path}/high-st-{args.mc}-comp-ref-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    if method == "refsplit":
                        statistics = np.load(
                            f"{stats_path}/high-st-{args.mc}-comp-refsplit-stats-{x}.npy",
                            allow_pickle=True,
                        )
                else:"""
                if method == "noref":
                    statistics = np.load(
                        f"{stats_path}/{args.mc}-comp-noref-stats-{x}.npy",
                        allow_pickle=True,
                    )
                if method == "ref":
                    statistics = np.load(
                        f"{stats_path}/{args.mc}-comp-ref-stats-{x}.npy",
                        allow_pickle=True,
                    )
                if method == "refsplit":
                    statistics = np.load(
                        f"{stats_path}/{args.mc}-comp-refsplit-stats-{x}.npy",
                        allow_pickle=True,
                    )
        except FileNotFoundError as e:
            print(f"Statistics file for {x} not found, skipping: {e}")
            continue

        obj = statistics.item()
        imc_transition_count_iters = obj["transitions_learned"]
        # stopping_threshold = obj["args"]["stopping_threshold"]

        imc_transition_counts[str(x)] = np.cumsum(imc_transition_count_iters).tolist()

        model_path = obj["args"]["model_path"]

        for y in trange(1, len(imc_transition_count_iters) + 1):
            initial_distribution = f"{model_path}-{y}-initial_interval.npy"
            transition_intervals = f"{model_path}-{y}-interval.npy"

            args.trans_path = transition_intervals
            args.init_path = initial_distribution

            transition_intervals, initial_distribution = load_imc(args)

            mon, mon_comps = create_monitor(
                transition_intervals,
                initial_distribution,
                "min",
                True,
                horizon,
                None,
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


def aggregated_stats_mc(
    coarse, stats_path, initial_amount, horizon, args, testing_samples
):
    mc_risks = {}

    mc_transition_counts = {}

    for x in range(1, 11):
        print(f"Experiment number {x}")
        try:
            if coarse:
                """if high_st:
                    if args.mc == " SnLw-10x10":
                        statistics = np.load(
                            f"{stats_path}/high-st-SnL-coarse-comp-mle-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    elif args.mc == "evadeV-6-3-coarse":
                        statistics = np.load(
                            f"{stats_path}/high-st-evadeV-6-3-coarse-comp-mle-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    else:
                        statistics = np.load(
                            f"{stats_path}/high-st-{args.mc}-coarse-comp-mle-stats-{x}.npy",
                            allow_pickle=True,
                        )
                else:"""
                if args.mc == "SnLw-10x10":
                    statistics = np.load(
                        f"{stats_path}/SnL-coarse-comp-mle-stats-{x}.npy",
                        allow_pickle=True,
                    )
                elif args.mc == "evadeV-6-3-coarse":
                    statistics = np.load(
                        f"{stats_path}/evadeV-6-3-coarse-comp-mle-stats-{x}.npy",
                        allow_pickle=True,
                    )
                else:
                    statistics = np.load(
                        f"{stats_path}/{args.mc}-coarse-comp-mle-stats-{x}.npy",
                        allow_pickle=True,
                    )
            else:
                """if high_st:
                    statistics = np.load(
                        f"{stats_path}/high-st-{args.mc}-comp-mle-stats-{x}.npy",
                        allow_pickle=True,
                    )
                else:"""
                statistics = np.load(
                    f"{stats_path}/{args.mc}-comp-mle-stats-{x}.npy",
                    allow_pickle=True,
                )
        except FileNotFoundError:
            print(f"Statistics file for {x} not found, skipping.")

            continue

        mc_sample_count = statistics["sample_counts"]

        mc_transition_count = []

        for a in mc_sample_count:
            mc_transition_count.append(a * (horizon + initial_amount))

        mc_transition_counts[str(x)] = mc_transition_count

        model_path = statistics["args"]["dump_model"]

        for y in trange(0, len(mc_transition_count)):
            model = f"{model_path}-{y}.pickl"

            with open(model, "rb") as file:
                data = pickle.load(file)

            model, observation_map, state_index_map = dict_to_pomdp(
                data[1], data[0], target_label=True, use_exact=True
            )

            monitor = create_mle_monitor(horizon, model)

            mc_risks[f"{x}-{y}"] = []

            for sample in testing_samples:
                subtrace = sample[:initial_amount]

                risk = test_monitor(
                    monitor,
                    [subtrace],
                    lambda x: observation_map[x],
                    skip_initial=True,
                    with_tqdm=False,
                )[subtrace]

                mc_risks[f"{x}-{y}"].append(float(risk))

    return mc_risks, mc_transition_counts


""" def distance_graph(
    high_st,
    coarse,
    target_risks,
    imc_risks,
    imc_transition_counts,
    mc_risks,
    mc_transition_counts,
    stopping_threshold,
    out_path,
): """


def fn_fp_comparison(coarse, alarms, imc_risks, mc_risks, target_risks, out_path):

    # iHMM
    imc_final_risks = {}

    ys = []
    for key in imc_risks.keys():
        ys.append(int(key.split("-")[1]))

    for key in imc_risks.keys():
        if key.split("-")[1] == str(max(ys)):
            imc_final_risks[key.split("-")[0]] = imc_risks[key]

    # HMM
    mc_final_risks = {}

    mc_ys = []
    for key in mc_risks.keys():
        mc_ys.append(int(key.split("-")[1]))

    for key in mc_risks.keys():
        if key.split("-")[1] == str(max(mc_ys)):
            mc_final_risks[key.split("-")[0]] = mc_risks[key]

    thresholds = [t / 1000 for t in range(0, 1001)]
    target_risks = np.array(target_risks, dtype=float)
    alarms = np.array(alarms)

    target_fnr = []
    target_fpr = []

    imc_fnr = {}
    imc_fpr = {}

    mc_fnr = {}
    mc_fpr = {}

    for key in imc_final_risks.keys():
        imc_fnr[key] = []
        imc_fpr[key] = []

    for key in mc_final_risks.keys():
        mc_fnr[key] = []
        mc_fpr[key] = []

    for t in thresholds:

        actual_positives = np.sum(alarms == 1)
        actual_negatives = np.sum(alarms == 0)
        total = len(alarms)

        # Target
        predictions_target = target_risks >= t

        fp_target = np.sum((predictions_target == 1) & (alarms == 0))
        fn_target = np.sum((predictions_target == 0) & (alarms == 1))

        fnr_target = fn_target / actual_positives if actual_positives > 0 else 0.0
        fpr_target = fp_target / actual_negatives if actual_negatives > 0 else 0.0

        target_fnr.append(fnr_target)
        target_fpr.append(fpr_target)

        for key in imc_final_risks.keys():

            predictions_imc = np.array(imc_final_risks[key], dtype=float) >= t

            fp_imc = np.sum((predictions_imc == True) & (alarms == 0))
            fn_imc = np.sum((predictions_imc == False) & (alarms == 1))

            fnr_imc = fn_imc / actual_positives if actual_positives > 0 else 0.0
            fpr_imc = fp_imc / actual_negatives if actual_negatives > 0 else 0.0

            imc_fnr[key].append(fnr_imc)
            imc_fpr[key].append(fpr_imc)

        for key in mc_final_risks.keys():

            predictions_mc = np.array(mc_final_risks[key], dtype=float) >= t

            fp_mc = np.sum((predictions_mc == 1) & (alarms == 0))
            fn_mc = np.sum((predictions_mc == 0) & (alarms == 1))

            fnr_mc = fn_mc / actual_positives if actual_positives > 0 else 0.0
            fpr_mc = fp_mc / actual_negatives if actual_negatives > 0 else 0.0

            mc_fnr[key].append(fnr_mc)
            mc_fpr[key].append(fpr_mc)

    fig, ax = plt.subplots(figsize=(8, 5))

    auc_fnr_target = np.trapz(target_fnr, thresholds)
    auc_fpr_target = np.trapz(target_fpr, thresholds)

    ax.plot(
        thresholds,
        target_fnr,
        label=f"Target FNR: (AUC {auc_fnr_target:.3f})",
        color="black",
        linestyle=":",
    )
    ax.plot(
        thresholds,
        target_fpr,
        label=f"Target FPR: (AUC {auc_fpr_target:.3f})",
        color="black",
        linestyle="--",
    )

    imc_FNRs = []
    imc_FPRs = []

    imc_fnr_aucs = []
    imc_fpr_aucs = []

    for key in imc_final_risks.keys():
        imc_FNRs.append(imc_fnr[key])
        imc_FPRs.append(imc_fpr[key])
        imc_fnr_aucs.append(np.trapz(imc_fnr[key], thresholds))
        imc_fpr_aucs.append(np.trapz(imc_fpr[key], thresholds))

    imc_FNRs = np.array(imc_FNRs)

    imc_fnr_mean = np.mean(imc_FNRs, axis=0)
    imc_fnr_std = np.std(imc_FNRs, axis=0)

    imc_fnr_aucs = np.array(imc_fnr_aucs)
    imc_fnr_auc = np.mean(imc_fnr_aucs, axis=0)

    imc_fpr_aucs = np.array(imc_fpr_aucs)
    imc_fpr_auc = np.mean(imc_fpr_aucs, axis=0)

    ax.plot(
        thresholds,
        imc_fnr_mean,
        label=f"iHMM mean FNR: (AUC {imc_fnr_auc:.3f})",
        color="red",
        linestyle=":",
    )

    plt.fill_between(
        thresholds,
        imc_fnr_mean - imc_fnr_std,
        imc_fnr_mean + imc_fnr_std,
        color="red",
        alpha=0.2,
    )

    imc_FPRs = np.array(imc_FPRs)

    imc_fpr_mean = np.mean(imc_FPRs, axis=0)
    imc_fpr_std = np.std(imc_FPRs, axis=0)

    ax.plot(
        thresholds,
        imc_fpr_mean,
        label=f"iHMM mean FPR: (AUC {imc_fpr_auc:.3f})",
        color="red",
        linestyle="--",
    )

    plt.fill_between(
        thresholds,
        imc_fpr_mean - imc_fpr_std,
        imc_fpr_mean + imc_fpr_std,
        color="red",
        alpha=0.2,
    )

    mc_FNRs = []
    mc_FPRs = []

    mc_fnr_aucs = []
    mc_fpr_aucs = []

    for key in mc_final_risks.keys():
        mc_FNRs.append(mc_fnr[key])
        mc_FPRs.append(mc_fpr[key])
        mc_fnr_aucs.append(np.trapz(mc_fnr[key], thresholds))
        mc_fpr_aucs.append(np.trapz(mc_fpr[key], thresholds))

    mc_FNRs = np.array(mc_FNRs)

    mc_fnr_mean = np.mean(mc_FNRs, axis=0)
    mc_fnr_std = np.std(mc_FNRs, axis=0)

    mc_fnr_aucs = np.array(mc_fnr_aucs)
    mc_fnr_auc = np.mean(mc_fnr_aucs, axis=0)

    mc_fpr_aucs = np.array(mc_fpr_aucs)
    mc_fpr_auc = np.mean(mc_fpr_aucs, axis=0)

    ax.plot(
        thresholds,
        mc_fnr_mean,
        label=f"HMM mean FNR: (AUC {mc_fnr_auc:.3f})",
        color="green",
        linestyle=":",
    )

    plt.fill_between(
        thresholds,
        mc_fnr_mean - mc_fnr_std,
        mc_fnr_mean + mc_fnr_std,
        color="green",
        alpha=0.2,
    )

    mc_FPRs = np.array(mc_FPRs)

    mc_fpr_mean = np.mean(mc_FPRs, axis=0)
    mc_fpr_std = np.std(mc_FPRs, axis=0)

    ax.plot(
        thresholds,
        mc_fpr_mean,
        label=f"HMM mean FPR: (AUC {mc_fpr_auc:.3f})",
        color="green",
        linestyle="--",
    )

    plt.fill_between(
        thresholds,
        mc_fpr_mean - mc_fpr_std,
        mc_fpr_mean + mc_fpr_std,
        color="green",
        alpha=0.2,
    )

    mc_fnr_mean = np.array(mc_fnr_mean)
    imc_fnr_mean = np.array(imc_fnr_mean)
    target_fnr = np.array(target_fnr)
    thresholds = np.array(thresholds)

    mask_2 = target_fnr < mc_fnr_mean
    mask_3 = target_fnr < imc_fnr_mean

    count = 0

    indices_2 = np.where(mask_2)[0]
    for k, g in groupby(enumerate(indices_2), lambda i: i[0] - i[1]):
        count += 1
        group = list(map(itemgetter(1), g))
        start = thresholds[group[0]]
        end = thresholds[group[-1]]
        if count < 1:
            ax.axvspan(
                start,
                end,
                color="peachpuff",
                alpha=0.5,
                label="HMM mean FNR > Target FNR",
            )
        else:
            ax.axvspan(start, end, color="peachpuff", alpha=0.5)

    indices_3 = np.where(mask_3)[0]
    for k, g in groupby(enumerate(indices_3), lambda i: i[0] - i[1]):
        count += 1
        group = list(map(itemgetter(1), g))
        start = thresholds[group[0]]
        end = thresholds[group[-1]]
        if count < 1:
            ax.axvspan(
                start,
                end,
                color="lightsteelblue",
                alpha=0.5,
                label="IHMM mean FNR > Target FNR",
            )
        else:
            ax.axvspan(start, end, color="lightsteelblue", alpha=0.5)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    if coarse:
        fig.savefig(
            f"{out_path}/rq_1_{args.mc}_coarse_FN_FP_multi.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            f"{out_path}/rq_1_{args.mc}_FN_FP_multi.pdf",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def distance_graph(
    coarse,
    target_risks,
    imc_risks,
    imc_transition_counts,
    imc_risks_ref,
    imc_transition_counts_ref,
    imc_risks_refsplit,
    imc_transition_counts_refsplit,
    out_path,
):

    # Extract unique experiment numbers from NO REF data
    imc_experiment_numbers = set()
    for key in imc_risks.keys():
        exp_num = int(key.split("-")[0])
        imc_experiment_numbers.add(exp_num)

    # Extract unique experiment numbers from REF data
    ref_experiment_numbers = set()
    for key in imc_risks_ref.keys():
        exp_num = int(key.split("-")[0])
        ref_experiment_numbers.add(exp_num)

    # Extract unique experiment numbers from REF with SPLITTING data
    refsplit_experiment_numbers = set()
    for key in imc_risks_refsplit.keys():
        exp_num = int(key.split("-")[0])
        refsplit_experiment_numbers.add(exp_num)

    # IMC
    distance_stats = {}

    for key in imc_risks.keys():
        total_distance = 0
        for x in range(len(target_risks)):
            # total_distance += testing_samples_weights[x] * abs(
            #    imc_risks[key][x] - target_risks[x]
            # )
            total_distance += abs(imc_risks[key][x] - target_risks[x])
        distance_stats[key] = total_distance / args.testing_samples

    distance_graph_data = {}
    for x in imc_experiment_numbers:
        distance_graph_data[x] = []

    for key in distance_stats.keys():
        exp_num = int(key.split("-")[0])
        if exp_num in imc_experiment_numbers:
            distance_graph_data[exp_num].append(distance_stats[key])

    final_distances = []
    for x in sorted(imc_experiment_numbers):
        if distance_graph_data[x]:  # Check if list is not empty
            final_distances.append(distance_graph_data[x][-1])

    graph_data = []
    for x in sorted(imc_experiment_numbers):
        if str(x) in imc_transition_counts and distance_graph_data[x]:
            graph_data.append([distance_graph_data[x], imc_transition_counts[str(x)]])

    # REF
    ref_distance_stats = {}

    for key in imc_risks_ref.keys():
        total_distance = 0
        for x in range(len(target_risks)):
            total_distance += abs(imc_risks_ref[key][x] - target_risks[x])
        ref_distance_stats[key] = total_distance / args.testing_samples

    ref_distance_graph_data = {}
    for x in ref_experiment_numbers:
        ref_distance_graph_data[x] = []

    for key in ref_distance_stats.keys():
        exp_num = int(key.split("-")[0])
        if exp_num in ref_experiment_numbers:
            ref_distance_graph_data[exp_num].append(ref_distance_stats[key])

    ref_final_distances = []
    for x in sorted(ref_experiment_numbers):
        if ref_distance_graph_data[x]:  # Check if list is not empty
            ref_final_distances.append(ref_distance_graph_data[x][-1])

    ref_graph_data = []
    for x in sorted(ref_experiment_numbers):
        if str(x) in imc_transition_counts_ref and ref_distance_graph_data[x]:
            ref_graph_data.append(
                [ref_distance_graph_data[x], imc_transition_counts_ref[str(x)]]
            )

    # REF with SPLITTING
    refsplit_distance_stats = {}

    for key in imc_risks_refsplit.keys():
        total_distance = 0
        for x in range(len(target_risks)):
            # total_distance += testing_samples_weights[x] * abs(
            #    imc_risks[key][x] - target_risks[x]
            # )
            total_distance += abs(imc_risks_refsplit[key][x] - target_risks[x])
        refsplit_distance_stats[key] = total_distance / args.testing_samples

    refsplit_distance_graph_data = {}
    for x in refsplit_experiment_numbers:
        refsplit_distance_graph_data[x] = []

    for key in refsplit_distance_stats.keys():
        exp_num = int(key.split("-")[0])
        if exp_num in refsplit_experiment_numbers:
            refsplit_distance_graph_data[exp_num].append(refsplit_distance_stats[key])

    refsplit_final_distances = []
    for x in sorted(refsplit_experiment_numbers):
        if refsplit_distance_graph_data[x]:  # Check if list is not empty
            refsplit_final_distances.append(refsplit_distance_graph_data[x][-1])

    refsplit_graph_data = []
    for x in sorted(refsplit_experiment_numbers):
        if str(x) in imc_transition_counts_refsplit and distance_graph_data[x]:
            refsplit_graph_data.append(
                [
                    refsplit_distance_graph_data[x],
                    imc_transition_counts_refsplit[str(x)],
                ]
            )

    log = False
    plt.figure()
    fig, ax = plt.subplots(figsize=(16, 8))

    # IMC AVERAGE PERFORMANCE
    transitions_data = []
    distance_data = []

    for entry in graph_data:
        transitions = entry[1]
        distance_daum = entry[0]

        transitions_data.append(transitions)
        distance_data.append(distance_daum)

    # Find common x range for interpolation
    min_x = min(min(transitions) for transitions in transitions_data)
    max_x = max(max(transitions) for transitions in transitions_data)
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
        color="red",
        label=f"No refinement, (Average final distance: {np.mean(final_distances):.3f})",
        linewidth=7,
        linestyle="--",
    )

    # Add shaded area for spread
    ax.fill_between(
        x_values,
        mean_distance - std_distance,
        mean_distance + std_distance,
        alpha=0.2,
        color="red",
    )

    # REF AVERAGE PERFORMANCE

    ref_transitions_data = []
    ref_distance_data = []

    for entry in ref_graph_data:
        transitions = entry[1]
        distance_daum = entry[0]

        ref_transitions_data.append(transitions)
        ref_distance_data.append(distance_daum)

    # Find common x range for interpolation
    min_x = min(min(transitions) for transitions in ref_transitions_data)
    max_x = max(max(transitions) for transitions in ref_transitions_data)
    ref_x_values = np.linspace(min_x, max_x, 500)

    # Interpolate all runs to common x values
    ref_interpolated_auc = []
    for auc, transitions in zip(ref_distance_data, ref_transitions_data):
        if log:
            auc = np.log10(auc)
        interpolated = np.interp(ref_x_values, transitions, auc)
        if log:
            interpolated = np.power(10, interpolated)
        ref_interpolated_auc.append(interpolated)

    # Calculate mean and std for interpolated y values
    ref_distance_array = np.array(ref_interpolated_auc)
    ref_mean_distance = np.mean(ref_distance_array, axis=0)
    ref_std_distance = np.std(ref_distance_array, axis=0)

    # Plot mean line
    ax.plot(
        ref_x_values,
        ref_mean_distance,
        color="blue",
        label=f"Refinement, (Average final distance: {np.mean(ref_final_distances):.3f})",
        linewidth=7,
        linestyle=":",
    )

    # Add shaded area for spread
    ax.fill_between(
        ref_x_values,
        ref_mean_distance - ref_std_distance,
        ref_mean_distance + ref_std_distance,
        alpha=0.2,
        color="blue",
    )

    # REF with SPLITTING AVERAGE PERFORMANCE
    refsplit_transitions_data = []
    refsplit_distance_data = []

    for entry in refsplit_graph_data:
        transitions = entry[1]
        distance_daum = entry[0]

        refsplit_transitions_data.append(transitions)
        refsplit_distance_data.append(distance_daum)

    # Find common x range for interpolation
    refsplit_min_x = min(min(transitions) for transitions in refsplit_transitions_data)
    refsplit_max_x = max(max(transitions) for transitions in refsplit_transitions_data)
    refsplit_x_values = np.linspace(refsplit_min_x, refsplit_max_x, 500)

    # Interpolate all runs to common x values
    refsplit_interpolated_auc = []
    for auc, transitions in zip(refsplit_distance_data, refsplit_transitions_data):
        if log:
            auc = np.log10(auc)
        interpolated = np.interp(refsplit_x_values, transitions, auc)
        if log:
            interpolated = np.power(10, interpolated)
        refsplit_interpolated_auc.append(interpolated)

    # Calculate mean and std for interpolated y values
    refsplit_distance_array = np.array(refsplit_interpolated_auc)
    refsplit_mean_distance = np.mean(refsplit_distance_array, axis=0)
    refsplit_std_distance = np.std(refsplit_distance_array, axis=0)

    # Plot mean line
    ax.plot(
        refsplit_x_values,
        refsplit_mean_distance,
        color="aqua",
        label=f"Refinement with splitting, (Average final distance: {np.mean(final_distances):.3f})",
        linewidth=7,
        linestyle="-.",
    )

    # Add shaded area for spread
    ax.fill_between(
        refsplit_x_values,
        refsplit_mean_distance - refsplit_std_distance,
        refsplit_mean_distance + refsplit_std_distance,
        alpha=0.2,
        color="aqua",
    )

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((4, 4))  # Force 10^4 scale
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="both", labelsize=25)
    ax.xaxis.get_offset_text().set_size(25)

    ax.set_xlabel("Explored states", fontsize=30)
    ax.set_ylabel("Distance to Target", fontsize=30)
    ax.legend(loc="upper right", fontsize=30)
    if log:
        plt.yscale("log")
    else:
        plt.ylim(bottom=0)
    ax.grid(True)
    plt.subplots_adjust(bottom=0.25)

    plt.tight_layout()
    if coarse:
        """if high_st:
            plt.savefig(
                f"{out_path}/rq_1_high-st-{args.mc}_coarse_distance_to_RRF.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        else:"""
        plt.savefig(
            f"{out_path}/rq_1_{args.mc}_coarse_distance_to_RRF.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        """if high_st:
            plt.savefig(
                f"{out_path}/rq_1_high-st-{args.mc}_distance_to_RRF.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        else:"""
        plt.savefig(
            f"{out_path}/rq_1_{args.mc}_distance_to_RRF.pdf",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def overestimation_graph(
    coarse,
    target_risks,
    imc_risks,
    imc_transition_counts,
    mc_risks,
    mc_transition_counts,
    out_path,
):

    plt.figure()
    plt.plot([0, 1], [0, 1], "--", color="black")

    mc_ys = []

    for key in mc_risks.keys():
        mc_ys.append(int(key.split("-")[1]))

    hmm_under = 0
    hmm_over = 0

    for key in mc_risks.keys():
        if key.split("-")[1] == str(max(mc_ys)):
            for x in range(len(target_risks)):
                if mc_risks[key][x] < target_risks[x]:
                    hmm_under += 1
                if mc_risks[key][x] >= target_risks[x]:
                    hmm_over += 1

    count_u = 0
    count_o = 0

    for key in mc_risks.keys():
        if key.split("-")[1] == str(max(mc_ys)):
            for x in range(len(target_risks)):
                if mc_risks[key][x] < target_risks[x]:
                    if count_u < 1:
                        plt.scatter(
                            mc_risks[key][x],
                            target_risks[x],
                            color="greenyellow",
                            marker="s",
                            s=3,
                            label=f"HMM under ({((hmm_under/(hmm_under + hmm_over))*100):.2f}%)",
                        )
                        count_u += 1
                    else:
                        plt.scatter(
                            mc_risks[key][x],
                            target_risks[x],
                            color="greenyellow",
                            marker="s",
                            s=3,
                        )
                if mc_risks[key][x] >= target_risks[x]:
                    if count_o < 1:
                        plt.scatter(
                            mc_risks[key][x],
                            target_risks[x],
                            color="darkgreen",
                            marker="s",
                            s=3,
                            label=f"HMM over ({((hmm_over/(hmm_under + hmm_over))*100):.2f}%)",
                        )
                        count_o += 1
                    else:
                        plt.scatter(
                            mc_risks[key][x],
                            target_risks[x],
                            color="darkgreen",
                            marker="s",
                            s=3,
                        )

    imc_ys = []

    for key in imc_risks.keys():
        imc_ys.append(int(key.split("-")[1]))

    ihmm_under = 0
    ihmm_over = 0

    for key in imc_risks.keys():
        if key.split("-")[1] == str(max(imc_ys)):
            for x in range(len(target_risks)):
                if imc_risks[key][x] < target_risks[x]:
                    ihmm_under += 1
                if imc_risks[key][x] >= target_risks[x]:
                    ihmm_over += 1

    count_u = 0
    count_o = 0

    for key in imc_risks.keys():
        if key.split("-")[1] == str(max(imc_ys)):
            for x in range(len(target_risks)):
                if imc_risks[key][x] < target_risks[x]:
                    if count_u < 1:
                        plt.scatter(
                            imc_risks[key][x],
                            target_risks[x],
                            color="orange",
                            marker="o",
                            s=3,
                            label=f"iHMM under ({((ihmm_under/(ihmm_under + ihmm_over))* 100):.2f} %)",
                        )
                        count_u += 1
                    else:
                        plt.scatter(
                            imc_risks[key][x],
                            target_risks[x],
                            color="orange",
                            marker="o",
                            s=3,
                        )
                if imc_risks[key][x] >= target_risks[x]:
                    if count_o < 1:
                        plt.scatter(
                            imc_risks[key][x],
                            target_risks[x],
                            color="red",
                            marker="o",
                            s=3,
                            label=f"iHMM over ({((ihmm_over / (ihmm_under + ihmm_over))*100):.2f}%)",
                        )
                        count_o += 1
                    else:
                        plt.scatter(
                            imc_risks[key][x],
                            target_risks[x],
                            color="red",
                            marker="o",
                            s=3,
                        )

    plt.legend(fontsize=15)

    plt.xlabel("iHMM and HMM risks", fontsize=15)
    plt.ylabel("Target risks", fontsize=15)
    plt.tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    if coarse:
        """if high_st:
            plt.savefig(
                f"{out_path}/rq_1_high-st{args.mc}_coarse_overestimation.pdf",
                dpi=300,
            )
        else:"""
        plt.savefig(
            f"{out_path}/rq_1_{args.mc}_coarse_overestimation.pdf",
            dpi=300,
        )
    else:
        """if high_st:
            plt.savefig(
                f"{out_path}/rq_1_high-st{args.mc}_overestimation.pdf",
                dpi=300,
            )
        else:"""
        plt.savefig(
            f"{out_path}/rq_1_{args.mc}_overestimation.pdf",
            dpi=300,
        )

    plt.show()


def main_imc(args: argparse.Namespace):
    setup_logging("rq1:" + args.mc)

    suo, initial_amount, horizon = build_suo(args)

    length = initial_amount + horizon

    testing_samples = []
    testing_samples_weights = []

    for x in range(args.testing_samples):
        path = suo.generate_random_traces_with_prob([], length)
        testing_samples.append(tuple(path[0][0]))
        testing_samples_weights.append(float(path[0][1]))

    if args.sys_vars != None:
        coarse = True
    elif args.mc == "SnLw-10x10":
        coarse = True
    elif args.mc == "evadeV-6-3-coarse":
        coarse = True
    else:
        coarse = False

    alarms = aggregted_alarms(testing_samples)

    target_risks = stats_true(
        horizon,
        initial_amount,
        testing_samples,
        suo,
    )

    imc_risks, imc_transition_counts = aggregated_stats_imc(
        "noref",
        coarse,
        args.stats_path,
        initial_amount,
        horizon,
        args,
        testing_samples,
    )

    mc_risks, mc_transition_counts = aggregated_stats_mc(
        coarse,
        args.stats_path,
        initial_amount,
        horizon,
        args,
        testing_samples,
    )

    if len(imc_risks) == 0 or len(mc_risks) == 0:
        print("No data for IMC or MC risks, skipping graph generation.")
        return

    overestimation_graph(
        coarse,
        target_risks,
        imc_risks,
        imc_transition_counts,
        mc_risks,
        mc_transition_counts,
        args.out,
    )

    fn_fp_comparison(
        coarse,
        alarms,
        imc_risks,
        mc_risks,
        target_risks,
        args.out,
    )

    imc_risks_ref, imc_transition_counts_ref = aggregated_stats_imc(
        "ref",
        coarse,
        args.stats_path,
        initial_amount,
        horizon,
        args,
        testing_samples,
    )

    imc_risks_refsplit, imc_transition_counts_refsplit = aggregated_stats_imc(
        "refsplit",
        coarse,
        args.stats_path,
        initial_amount,
        horizon,
        args,
        testing_samples,
    )

    distance_graph(
        coarse,
        target_risks,
        imc_risks,
        imc_transition_counts,
        imc_risks_ref,
        imc_transition_counts_ref,
        imc_risks_refsplit,
        imc_transition_counts_refsplit,
        args.out,
    )

    test_data = {}
    test_data["model"] = args.mc
    test_data["coarse"] = coarse
    test_data["testing_samples"] = testing_samples
    test_data["testing_samples_amount"] = args.testing_samples
    test_data["alarms"] = alarms
    test_data["target_risks"] = target_risks
    test_data["mc_risks"] = mc_risks
    test_data["mc_transition_counts"] = mc_transition_counts
    test_data["imc_risks"] = imc_risks
    test_data["imc_transition_counts"] = imc_transition_counts
    test_data["imc_risks_ref"] = imc_risks_ref
    test_data["imc_transition_counts_ref"] = imc_transition_counts_ref
    test_data["imc_risks_refsplit"] = imc_risks_refsplit
    test_data["imc_transition_counts_refsplit"] = imc_transition_counts_refsplit

    if coarse == True:
        file_name = os.path.join(args.out, f"testdata_rq_1_{args.mc}_coarse.pkl")
    else:
        file_name = os.path.join(args.out, f"testdata_rq_1_{args.mc}.pkl")

    # Save dictionary
    with open(file_name, "wb") as f:
        pickle.dump(test_data, f)


def build_learning_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Learning Parameters")

    group.add_argument(
        "-s",
        "--testing-samples",
        type=int,
        default=500,
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

    parser.add_argument("--stats-path", type=str, help="Path stats")

    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="out/results",
        help="Output path for the results",
    )

    return parser


if __name__ == "__main__":
    parser = testing_argsparser()
    args = parser.parse_args()
    main_imc(args)

# See scripts/analysis.sh for the example commands


# python -m premise.interval.rq_1 --mc evadeV-5-3 --stats-path /workspaces/premise/out/stats/2025-08-01_08-37-30 --out /workspaces/premise/premise/analysis
