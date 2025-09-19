import argparse
import glob
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle
from sklearn import metrics
import matplotlib.ticker as ticker
from itertools import accumulate


from premise.interval.model_free.regression_model import prep_traces_onehot_encoder, create_onehot_encoder
from premise.interval.utils import setup_logging
from premise.interval.conformal_prediction.train_stoch_seq_nsc import *
from premise.interval.conformal_prediction.train_seq_se import *
from premise.interval.conformal_prediction.train_seq_nsc import *
from premise.interval.conformal_prediction.CP_Classification import *
from premise.interval.conformal_prediction.CP_Regression import *
from premise.interval.conformal_prediction.SeqDataset import *
import torch
from torch.autograd import Variable
import premise.interval.conformal_prediction.utility_functions as utils
import numpy as np
import argparse
from premise.interval.conformal_prediction.InvertedPendulum import *
from premise.interval.conformal_prediction.MC_model import *
import torch.nn.functional

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


def stats_true(horizon, initial_amount, testing_samples, suo):

    mon = suo.create_target_monitor()
    target_risks = []

    for trace in tqdm(testing_samples):
        sub_trace: Trace = trace[:initial_amount]

        target_risk = test_monitor(
            mon,
            [sub_trace],
            with_tqdm=False,
        )[sub_trace]

        target_risks.append(float(target_risk))

    return target_risks


def aggregted_alarms(testing_samples):
    alarms = []

    for trace in tqdm(testing_samples):
        alarms.append(any([s[2] for s in trace]))

    alarms = np.array(alarms).astype(int)

    return alarms


def aggregated_stats_imc(
    high_st,
    coarse,
    method,
    mc,
    stats_path,
    initial_amount,
    horizon,
    args,
    testing_samples,
):

    imc_risks = {}

    imc_transition_counts = {}
    imc_distances = {}
    stopping_threshold = None

    for x in range(1, 11):
        print(f"Experiment number {x}")
        try:
            if coarse:
                if high_st:
                    if method == "noref":
                        if args.mc == "SnLw-10x10":
                            statistics = np.load(
                                f"{stats_path}/high-st-SnL-coarse_norefinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "evadeV-6-3-coarse":
                            statistics = np.load(
                                f"{stats_path}/high-st-evadeV-6-3-coarse_norefinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "SnL-10x10":
                            statistics = np.load(
                                f"{stats_path}/high-st-SnL-coarse-noref-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        else:
                            statistics = np.load(
                                f"{stats_path}/high-st-{mc}-coarse_norefinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                    elif method == "ref":
                        if args.mc == "SnLw-10x10":
                            statistics = np.load(
                                f"{stats_path}/high-st-SnL-coarse_refinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "evadeV-6-3-coarse":
                            statistics = np.load(
                                f"{stats_path}/high-st-evadeV-6-3-coarse_refinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "SnL-10x10":
                            statistics = np.load(
                                f"{stats_path}/high-st-SnL-coarse-ref-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        else:
                            statistics = np.load(
                                f"{stats_path}/high-st-{mc}-coarse_refinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                    elif method == "refsplit":
                        if args.mc == "SnLw-10x10":
                            statistics = np.load(
                                f"{stats_path}/high-st-SnL-coarse_refsplitinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "evadeV-6-3-coarse":
                            statistics = np.load(
                                f"{stats_path}/high-st-evadeV-6-3-coarse_refsplitinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "SnL-10x10":
                            statistics = np.load(
                                f"{stats_path}/high-st-SnL-coarse-refsplit-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        else:
                            statistics = np.load(
                                f"{stats_path}/high-st-{mc}-coarse_refsplitinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                else:
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
                                f"{stats_path}/{mc}-coarse_norefinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                    elif method == "ref":
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
                                f"{stats_path}/{mc}-coarse_refinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                    elif method == "refsplit":
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
                                f"{stats_path}/{mc}-coarse_refsplitinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
            else:
                if high_st:
                    if method == "noref":
                        statistics = np.load(
                            f"{stats_path}/high-st-{mc}-comp-noref-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    elif method == "ref":
                        statistics = np.load(
                            f"{stats_path}/high-st-{mc}-comp-ref-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    elif method == "refsplit":
                        statistics = np.load(
                            f"{stats_path}/high-st-{mc}-comp-refsplit-stats-{x}.npy",
                            allow_pickle=True,
                        )
                else:
                    if method == "noref":
                        statistics = np.load(
                            f"{stats_path}/{mc}-comp-noref-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    elif method == "ref":
                        statistics = np.load(
                            f"{stats_path}/{mc}-comp-ref-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    elif method == "refsplit":
                        statistics = np.load(
                            f"{stats_path}/{mc}-comp-refsplit-stats-{x}.npy",
                            allow_pickle=True,
                        )
        except FileNotFoundError:
            print(f"Statistics file for {x} not found, skipping.")
            logger.warning(f"Statistics file for {x} not found, skipping.")
            continue

        obj = statistics.item()
        imc_transition_count = obj["transitions_learned"]
        stopping_threshold = obj["args"]["stopping_threshold"]
        distances = obj["distances"]
        imc_distances[str(x)] = distances
        imc_transition_counts[str(x)] = imc_transition_count

        model_path = obj["args"]["model_path"]

        for y in range(1, len(imc_transition_count) + 1):
            args.init_path = f"{model_path}-{y}-initial_interval.npy"
            args.trans_path = f"{model_path}-{y}-interval.npy"

            transition_intervals, initial_distribution = load_imc(args)

            # Build the premise monitor on the learned model
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

    return imc_risks, imc_transition_counts, stopping_threshold, imc_distances


def aggregated_stats_regression(
    high_st,
    coarse,
    mc,
    model_path,
    stats_path,
    testing_samples,
    horizon,
    initial_amount,
):

    regression_risks = {}

    for x in range(1, 11):
        regression_risks[f"{x}"] = []

        try:
                if coarse:
                    if high_st:
                        if args.mc == "SnLw-10x10":
                            statistics = np.load(
                                f"{stats_path}/high-st-SnL-coarse-comp-reg-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "evadeV-6-3-coarse":
                            statistics = np.load(
                                f"{stats_path}/high-st-evadeV-6-3-coarse-comp-reg-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "SnL-10x10":
                            statistics = np.load(
                                f"{stats_path}/high-st-SnL-coarse-comp-reg-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        else:
                            statistics = np.load(
                                f"{stats_path}/high-st-{mc}-coarse-comp-reg-stats-{x}.npy",
                                allow_pickle=True,
                            )
                    else:
                        if args.mc == "SnLw-10x10":
                            statistics = np.load(
                                f"{stats_path}/SnL-coarse-comp-reg-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "evadeV-6-3-coarse":
                            statistics = np.load(
                                f"{stats_path}/evadeV-6-3-coarse-comp-reg-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        else:
                            statistics = np.load(
                                f"{stats_path}/{mc}-coarse-comp-reg-stats-{x}.npy",
                                allow_pickle=True,
                            )
                else:
                    if high_st:
                        statistics = np.load(
                            f"{stats_path}/high-st-{mc}-comp-reg-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    else:
                        statistics = np.load(
                            f"{stats_path}/{mc}-comp-reg-stats-{x}.npy",
                            allow_pickle=True,
                        )

        except FileNotFoundError:
                print(f"Statistics file for {x} not found, skipping.")
                continue

        obj = statistics.item()
        observations = obj["observations"]
        model_path = obj["args"]["model_path"]
        ohe = obj["one_hot_encoder"]

      
        reg_model = np.load(f"{model_path}.npy", allow_pickle=True).item()

        column_names = [
                f"Step{s}_Obs{o}" for s in range(initial_amount) for o in observations
        ]

        reg_sub_trace = prep_traces_onehot_encoder(
                    ohe, testing_samples, initial_amount)
        X = prep_traces_onehot_encoder(ohe, testing_samples, initial_amount)
        regression_risks[f'{x}'] = reg_model.predict(X)

    return regression_risks


def aggreagted_stats_conformal(high_st, new_noisy, coarse, mc, model_path, stats_path):

    conformal_risks = {}
    #conformal_ys = {}

    for x in range(1, 11):
        try:
            if coarse:
                if high_st:
                    paths = glob.glob(
                        f"{model_path}/high-st-{mc}_coarse_comp_conformal_pred_state_estimator_{x}_*.pt"
                    )
                else:
                    paths = glob.glob(
                        f"{model_path}/{mc}_coarse_comp_conformal_pred_state_estimator_{x}_*.pt"
                    )
            else:
                if high_st:
                    paths = glob.glob(
                        f"{model_path}/high-st-{mc}_comp_conformal_pred_state_estimator_{x}_*.pt"
                    )
                else:
                    paths = glob.glob(
                        f"{model_path}/{mc}_comp_conformal_pred_state_estimator_{x}_*.pt"
                    )
        except FileNotFoundError:
            print(f"Statistics file for {x} not found, skipping.")
            continue

        #conformal_ys[str(x)] = []

        #for path in paths:
        #    match = re.search(r"_(\d+)\.pt$", path)
        #    if match:
        #        conformal_ys[str(x)].append(int(match.group(1)))

        #for key in conformal_ys.keys():
        #    conformal_ys[key].sort()

        #for y in conformal_ys[str(x)]:

        conformal_risks[f"{x}"] = []
        try:
                if coarse:
                    if high_st:
                            print(f"{model_path}/high-st-{mc}_coarse_comp_conformal_pred_state_estimator_{x-1}_*.pt")
                            print('/workspaces/premise/out/models/2025-09-16_07-48-07/high-st-SnL-10x10_coarse_comp_conformal_pred_state_estimator_0_3775.pt')
                            state_estimator = torch.load(
                                glob.glob(f"{model_path}/high-st-{mc}_coarse_comp_conformal_pred_state_estimator_{x-1}_*.pt")[0],
                                weights_only=False,
                            )
                            label_estimator = torch.load(
                                glob.glob(f"{model_path}/high-st-{mc}_coarse_comp_conformal_pred_label_estimator_{x-1}_*.pt")[0],
                                weights_only=False,
                            )
                            cp_classification = torch.load(
                                glob.glob(f"{model_path}/high-st-{mc}_coarse_comp_conformal_pred_cp_classification_{x-1}_*.pt")[0],
                                weights_only=False,
                            )

                            with open(
                                glob.glob(f"{stats_path}/high-st-{mc}_coarse_comp_conformal_pred_rejection_classifier_{x-1}_*.pickle")[0],
                                "rb",
                            ) as f:
                                rej_classifier = pickle.load(f)

                            rejection_classifier = rej_classifier["rej_rule"]

                            with open(
                                glob.glob(f"{stats_path}/high-st-{mc}_coarse_comp_conformal_pred_conformal_stats_{x-1}_*.pickle")[0],
                                "rb",
                            ) as f:
                                conformal_stats = pickle.load(f)
                    else:
                        state_estimator = torch.load(
                            f"{model_path}/{mc}_coarse_comp_conformal_pred_state_estimator_{x}_{y}.pt",
                            weights_only=False,
                        )
                        label_estimator = torch.load(
                            f"{model_path}/{mc}_coarse_comp_conformal_pred_label_estimator_{x}_{y}.pt",
                            weights_only=False,
                        )
                        cp_classification = torch.load(
                            f"{model_path}/{mc}_coarse_comp_conformal_pred_cp_classification_{x}_{y}.pt",
                            weights_only=False,
                        )

                        with open(
                            f"{stats_path}/{mc}_coarse_comp_conformal_pred_rejection_classifier_{x}_{y}.pickle",
                            "rb",
                        ) as f:
                            rej_classifier = pickle.load(f)

                        rejection_classifier = rej_classifier["rej_rule"]

                        with open(
                            f"{stats_path}/{mc}_coarse_comp_conformal_pred_conformal_stats_{x}_{y}.pickle",
                            "rb",
                        ) as f:
                            conformal_stats = pickle.load(f)

                else:
                    if high_st:
                        state_estimator = torch.load(
                            f"{model_path}/high-st-{mc}_comp_conformal_pred_state_estimator_{x}_{y}.pt",
                            weights_only=False,
                        )
                        label_estimator = torch.load(
                            f"{model_path}/high-st-{mc}_comp_conformal_pred_label_estimator_{x}_{y}.pt",
                            weights_only=False,
                        )
                        cp_classification = torch.load(
                            f"{model_path}/high-st-{mc}_comp_conformal_pred_cp_classification_{x}_{y}.pt",
                            weights_only=False,
                        )

                        with open(
                            f"{stats_path}/high-st-{mc}_comp_conformal_pred_rejection_classifier_{x}_{y}.pickle",
                            "rb",
                        ) as f:
                            rej_classifier = pickle.load(f)

                        rejection_classifier = rej_classifier["rej_rule"]

                        with open(
                            f"{stats_path}/high-st-{mc}_comp_conformal_pred_conformal_stats_{x}_{y}.pickle",
                            "rb",
                        ) as f:
                            conformal_stats = pickle.load(f)
                    else:
                        state_estimator = torch.load(
                            f"{model_path}/{mc}_comp_conformal_pred_state_estimator_{x}_{y}.pt",
                            weights_only=False,
                        )
                        label_estimator = torch.load(
                            f"{model_path}/{mc}_comp_conformal_pred_label_estimator_{x}_{y}.pt",
                            weights_only=False,
                        )
                        cp_classification = torch.load(
                            f"{model_path}/{mc}_comp_conformal_pred_cp_classification_{x}_{y}.pt",
                            weights_only=False,
                        )

                        with open(
                            f"{stats_path}/{mc}_comp_conformal_pred_rejection_classifier_{x}_{y}.pickle",
                            "rb",
                        ) as f:
                            rej_classifier = pickle.load(f)

                        rejection_classifier = rej_classifier["rej_rule"]

                        with open(
                            f"{stats_path}/{mc}_comp_conformal_pred_conformal_stats_{x}_{y}.pickle",
                            "rb",
                        ) as f:
                            conformal_stats = pickle.load(f)

        except FileNotFoundError:
                print(f"Statistics file for {x} not found, skipping.")
                continue

        new_noisy_scaled = -1 + 2 * (
                new_noisy - conformal_stats["dataset.MIN[1]"]
            ) / (conformal_stats["dataset.MAX[1]"] - conformal_stats["dataset.MIN[1]"])
        Y1 = np.transpose(new_noisy_scaled, (0, 2, 1))
        Y1t = Variable(FloatTensor(Y1))

        state_estimator.eval()
        state_estim = state_estimator(Y1t)
        label_estimator.eval()
        label_hypothesis = label_estimator(state_estim)

        label_prob = torch.nn.functional.softmax(label_hypothesis, dim=1)
        error_prob = label_prob[:, 1]
        error_prob = error_prob.tolist()

        pool_conf_cred = cp_classification.compute_confidence_credibility(
                np.transpose(new_noisy_scaled, (0, 2, 1))
        )
        keep_mask = utils.apply_svc_query_strategy(
                rejection_classifier, pool_conf_cred
        )

        for u in range(len(error_prob)):
                if keep_mask[u] == -1.0:
                    error_prob[u] = 1.0

        for u in range(len(error_prob)):
                conformal_risks[f"{x}"].append(error_prob[u])

    return conformal_risks


def roc_curve_model_based(
    coarse,
    current_threashold,
    alarms,
    imc_risks,
    imc_risks_ref,
    target_risks,
    imc_distances,
    imc_distances_ref,
    imc_risks_ref_splitting,
    imc_distances_ref_splitting,
    out_path,
):

    # Extract unique experiment numbers from IMC data
    imc_experiment_numbers = set()
    for key in imc_risks.keys():
        exp_num = int(key.split("-")[0])
        imc_experiment_numbers.add(exp_num)

    # Extract unique experiment numbers from IMC data
    imc_ref_experiment_numbers = set()
    for key in imc_risks_ref.keys():
        exp_num = int(key.split("-")[0])
        imc_ref_experiment_numbers.add(exp_num)

    # Extract unique experiment numbers from IMC data
    imc_ref_split_experiment_numbers = set()
    for key in imc_risks_ref_splitting.keys():
        exp_num = int(key.split("-")[0])
        imc_ref_split_experiment_numbers.add(exp_num)

    # No Refinement
    imc_final_risks = {}

    ys = []
    for key in imc_risks.keys():
        ys.append(int(key.split("-")[1]))

    for key in imc_risks.keys():
        if key.split("-")[1] == str(max(ys)):
            imc_final_risks[key.split("-")[0]] = imc_risks[key]

    # Refinement
    imc_ref_final_risks = {}

    ys_refinement  = []
    for key in imc_risks_ref.keys():
        ys_refinement.append(int(key.split("-")[1]))

    for key in imc_risks_ref.keys():
        if key.split("-")[1] == str(max(ys_refinement)):
            imc_ref_final_risks[key.split("-")[0]] = imc_risks_ref[key]

    # Refinement with splitting
    imc_ref_split_final_risks = {}

    ys_splitting = []
    for key in imc_risks_ref_splitting.keys():
        ys_splitting.append(int(key.split("-")[1]))

    for key in imc_risks_ref_splitting.keys():
        if key.split("-")[1] == str(max(ys_splitting)):
            imc_ref_split_final_risks[key.split("-")[0]] = imc_risks_ref_splitting[key]

    plt.figure()
    fig, ax = plt.subplots(figsize=(16, 12))

    # REFINEMENT WITH SPLITTING MEAN PERFORMANCE
    ref_splitting_imc_roc_data_mean = {}

    for key in imc_ref_split_final_risks.keys():

        fpr, tpr, thresholds = metrics.roc_curve(alarms, imc_ref_split_final_risks[key])
        roc_auc = metrics.auc(fpr, tpr)
        ref_splitting_imc_roc_data_mean[key] = [fpr, tpr, roc_auc]

    mean_fpr = np.linspace(0, 1, 100)

    tprs = []
    aucs = []

    for key in ref_splitting_imc_roc_data_mean.keys():
        interp_tpr = np.interp(
            mean_fpr,
            ref_splitting_imc_roc_data_mean[key][0],
            ref_splitting_imc_roc_data_mean[key][1],
        )
        aucs.append(ref_splitting_imc_roc_data_mean[key][2])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="aqua",
        label=f"Refinement with splitting, (Mean AUC = {mean_auc:.2f})",
        linewidth=5,
        linestyle="-.",
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="blue",
        alpha=0.2,
    )

    # REFINEMENT MEAN PERFORMANCE
    ref_imc_roc_data_mean = {}

    for key in imc_ref_final_risks.keys():

        fpr, tpr, thresholds = metrics.roc_curve(alarms, imc_ref_final_risks[key])
        roc_auc = metrics.auc(fpr, tpr)
        ref_imc_roc_data_mean[key] = [fpr, tpr, roc_auc]

    mean_fpr = np.linspace(0, 1, 100)

    tprs = []
    aucs = []

    for key in ref_imc_roc_data_mean.keys():
        interp_tpr = np.interp(
            mean_fpr, ref_imc_roc_data_mean[key][0], ref_imc_roc_data_mean[key][1]
        )
        aucs.append(ref_imc_roc_data_mean[key][2])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="blue",
        label=f"Refinement, (Mean AUC = {mean_auc:.2f})",
        linewidth=5,
        linestyle=":",
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="blue",
        alpha=0.2,
    )

    # NO REFINEMENT MEAN PERFORMANCE
    imc_roc_data_mean = {}

    for key in imc_final_risks.keys():
        fpr, tpr, thresholds = metrics.roc_curve(alarms, imc_final_risks[key])
        roc_auc = metrics.auc(fpr, tpr)
        imc_roc_data_mean[key] = [fpr, tpr, roc_auc]

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for key in imc_roc_data_mean.keys():
        interp_tpr = np.interp(
            mean_fpr, imc_roc_data_mean[key][0], imc_roc_data_mean[key][1]
        )
        aucs.append(imc_roc_data_mean[key][2])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="red",
        label=f"No Refinement, (Mean AUC = {mean_auc:.2f})",
        linewidth=5,
        linestyle="--",
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="red",
        alpha=0.2,
    )

    fpr, tpr, threshold = metrics.roc_curve(alarms, target_risks)
    roc_auc = metrics.auc(fpr, tpr)
    target_auc = roc_auc

    ax.plot(
        fpr,
        tpr,
        color="black",
        label=f"Target Monitor, AUC = {target_auc:.2f}",
        linewidth=5,
        linestyle="-",
        marker="D",
    )

    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel("True Positive Rate", fontsize=35)
    plt.xlabel("False Positive Rate", fontsize=35)
    plt.legend(loc="lower right", fontsize=30)
    plt.tick_params(axis="both")
    plt.grid(True)
    plt.tight_layout()

    if coarse:
        plt.savefig(
            f"{out_path}/rq2_{args.mc}_coarse_ROC_threashold_comparison_{current_threashold}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            f"{out_path}/rq2_{args.mc}_ROC_threashold_comparison_{current_threashold}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def plot_roc_curve(
    stopping_threshold,
    high_st,
    coarse,
    alarms,
    imc_risks_ref,
    imc_risks_ref_splitting,
    regression_risks,
    conformal_risks,
    target_risks,
    out_path,
):

    # Refinement
    imc_ref_final_risks = {}

    ys = []
    for key in imc_risks_ref.keys():
        ys.append(int(key.split("-")[1]))

    for key in imc_risks_ref.keys():
        if key.split("-")[1] == str(max(ys)):
            imc_ref_final_risks[key.split("-")[0]] = imc_risks_ref[key]

    # Refinement with splitting
    imc_ref_split_final_risks = {}

    ys_splitting = []
    for key in imc_risks_ref_splitting.keys():
        ys_splitting.append(int(key.split("-")[1]))

    for key in imc_risks_ref_splitting.keys():
        if key.split("-")[1] == str(max(ys_splitting)):
            imc_ref_split_final_risks[key.split("-")[0]] = imc_risks_ref_splitting[key]

    # Regression
    reg_final_risks = {}

    #for x in range(1, 11):
    for key in regression_risks.keys():
        #if int(key.split("-")[1]) == max(regression_ys[str(x)]):
        reg_final_risks[key] = regression_risks[key]

    # Confromal Prediction
    conformal_final_risks = {}

    #for x in range(1, 11):
    for key in conformal_risks.keys():
        #if int(key.split("-")[1]) == max(conformal_ys[str(x)]):
        conformal_final_risks[key] = conformal_risks[key]

    plt.figure()
    fig, ax = plt.subplots(figsize=(16, 12))

    # REFINEMENT MEAN PERFORMANCE
    ref_imc_roc_data = {}

    for key in imc_ref_final_risks.keys():

        fpr, tpr, thresholds = metrics.roc_curve(alarms, imc_ref_final_risks[key])
        roc_auc = metrics.auc(fpr, tpr)
        ref_imc_roc_data[key] = [fpr, tpr, roc_auc]

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for key in ref_imc_roc_data.keys():
        interp_tpr = np.interp(
            mean_fpr, ref_imc_roc_data[key][0], ref_imc_roc_data[key][1]
        )
        aucs.append(ref_imc_roc_data[key][2])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="blue",
        label=f"Refinement, (Mean AUC = {mean_auc:.2f})",
        linewidth=5,
        linestyle=":",
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="blue",
        alpha=0.2,
    )

    # REFINEMENT WITH SPLITTING MEAN PERFORMANCE
    ref_splitting_imc_roc_data = {}

    for key in imc_ref_split_final_risks.keys():

        fpr, tpr, thresholds = metrics.roc_curve(alarms, imc_ref_split_final_risks[key])
        roc_auc = metrics.auc(fpr, tpr)
        ref_splitting_imc_roc_data[key] = [fpr, tpr, roc_auc]

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for key in ref_splitting_imc_roc_data.keys():
        interp_tpr = np.interp(
            mean_fpr,
            ref_splitting_imc_roc_data[key][0],
            ref_splitting_imc_roc_data[key][1],
        )
        aucs.append(ref_splitting_imc_roc_data[key][2])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="aqua",
        label=f"Refinement with splitting, (Mean AUC = {mean_auc:.2f})",
        linewidth=5,
        linestyle="-.",
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="aqua",
        alpha=0.2,
    )

    # REGRESSION MEAN PERFORMANCE
    regression_roc_data = {}

    for key in reg_final_risks.keys():

        fpr, tpr, thresholds = metrics.roc_curve(alarms, reg_final_risks[key])
        roc_auc = metrics.auc(fpr, tpr)
        regression_roc_data[key] = [fpr, tpr, roc_auc]

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for key in regression_roc_data.keys():
        interp_tpr = np.interp(
            mean_fpr, regression_roc_data[key][0], regression_roc_data[key][1]
        )
        aucs.append(regression_roc_data[key][2])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="green",
        label=f"Regression, (Mean AUC = {mean_auc:.2f})",
        linewidth=5,
        linestyle="--",
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="green",
        alpha=0.2,
    )

    # CONFORMAL MEAN PERFORMANCE
    conformal_roc_data = {}

    for key in conformal_final_risks.keys():

        fpr, tpr, thresholds = metrics.roc_curve(alarms, conformal_final_risks[key])
        roc_auc = metrics.auc(fpr, tpr)
        conformal_roc_data[key] = [fpr, tpr, roc_auc]

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for key in conformal_roc_data.keys():
        interp_tpr = np.interp(
            mean_fpr, conformal_roc_data[key][0], conformal_roc_data[key][1]
        )
        aucs.append(conformal_roc_data[key][2])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="orange",
        label=f"Conformal Prediction, (Mean AUC = {mean_auc:.2f})",
        linewidth=5,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="orange",
        alpha=0.2,
    )

    fpr, tpr, threshold = metrics.roc_curve(alarms, target_risks)
    roc_auc = metrics.auc(fpr, tpr)
    target_auc = roc_auc

    ax.plot(
        fpr,
        tpr,
        color="black",
        label=f"Target Monitor, AUC = {target_auc:.2f}",
        linewidth=5,
        linestyle="-",
        marker="D",
    )

    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel("True Positive Rate", fontsize=35)
    plt.xlabel("False Positive Rate", fontsize=35)
    plt.legend(loc="lower right", fontsize=30)
    plt.tick_params(axis="both")
    plt.grid(True)
    plt.tight_layout()

    if coarse:
        if high_st:
            plt.savefig(
                f"{out_path}/rq_3_high-st-{args.mc}_coarse_model_based_model_free_ROC.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                f"{out_path}/rq_3_{args.mc}_coarse_model_based_model_free_ROC.pdf",
                dpi=300,
                bbox_inches="tight",
            )
    else:
        if high_st:
            plt.savefig(
                f"{out_path}/rq_3_high-st-{args.mc}_model_based_model_free_ROC.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                f"{out_path}/rq_3_{args.mc}_model_based_model_free_ROC.pdf",
                dpi=300,
                bbox_inches="tight",
            )

    plt.show()


def auc_graph_prep(
    imc_risks,
    target_risks,
    alarms,
    imc_risks_ref,
    imc_transition_counts_ref,
    imc_risks_ref_splitting,
    imc_transition_counts_ref_splitting,
):

    # Extract unique experiment numbers from IMC data
    imc_experiment_numbers = set()
    for key in imc_risks.keys():
        exp_num = int(key.split("-")[0])
        imc_experiment_numbers.add(exp_num)

    # Extract unique experiment numbers from IMC data
    imc_ref_experiment_numbers = set()
    for key in imc_risks_ref.keys():
        exp_num = int(key.split("-")[0])
        imc_ref_experiment_numbers.add(exp_num)

    # Extract unique experiment numbers from IMC data
    imc_ref_split_experiment_numbers = set()
    for key in imc_risks_ref_splitting.keys():
        exp_num = int(key.split("-")[0])
        imc_ref_split_experiment_numbers.add(exp_num)

    fpr, tpr, threshold = metrics.roc_curve(alarms, target_risks)
    roc_auc = metrics.auc(fpr, tpr)
    target_auc = roc_auc

    imc_auc = {}

    for imc_key in imc_risks.keys():
        fpr, tpr, threshold = metrics.roc_curve(alarms, imc_risks[imc_key])
        roc_auc = metrics.auc(fpr, tpr)
        imc_auc[imc_key] = roc_auc

    imc_ref_auc = {}

    for imc_ref_key in imc_risks_ref.keys():
        fpr, tpr, threshold = metrics.roc_curve(alarms, imc_risks_ref[imc_ref_key])
        roc_auc = metrics.auc(fpr, tpr)
        imc_ref_auc[imc_ref_key] = roc_auc

    imc_ref_splitting_auc = {}

    for imc_ref_key_splitting in imc_risks_ref_splitting.keys():
        fpr, tpr, threshold = metrics.roc_curve(
            alarms, imc_risks_ref_splitting[imc_ref_key_splitting]
        )
        roc_auc = metrics.auc(fpr, tpr)
        imc_ref_splitting_auc[imc_ref_key_splitting] = roc_auc

    # reg_auc = {}

    # for reg_key in regression_risks.keys():
    #     fpr, tpr, threshold = metrics.roc_curve(alarms, regression_risks[reg_key])
    #     roc_auc = metrics.auc(fpr, tpr)
    #     reg_auc[reg_key] = roc_auc

    # conformal_auc = {}

    # for conformal_key in conformal_risks.keys():
    #     fpr, tpr, threshold = metrics.roc_curve(alarms, conformal_risks[conformal_key])
    #     roc_auc = metrics.auc(fpr, tpr)
    #     conformal_auc[conformal_key] = roc_auc

    imc_results = {}

    for x in imc_experiment_numbers:
        imc_results[str(x)] = []

    for key in imc_auc.keys():
        for entry in imc_results.keys():
            x = key.split("-")[0]
            if x == entry:
                imc_results[x].append(imc_auc[key])

    imc_ref_results = {}

    for x in imc_ref_experiment_numbers:
        imc_ref_results[str(x)] = []

    for key in imc_ref_auc.keys():
        for entry in imc_ref_results.keys():
            x = key.split("-")[0]
            if x == entry:
                imc_ref_results[x].append(imc_ref_auc[key])

    imc_ref_splitting_results = {}

    for x in imc_ref_split_experiment_numbers:
        imc_ref_splitting_results[str(x)] = []

    for key in imc_ref_splitting_auc.keys():
        for entry in imc_ref_splitting_results.keys():
            x = key.split("-")[0]
            if x == entry:
                imc_ref_splitting_results[x].append(imc_ref_splitting_auc[key])

    # reg_results = {}

    # for x in range(1,11):
    #     reg_results[str(x)] = []

    # for key in reg_auc.keys():
    #     for entry in reg_results.keys():
    #         x = key.split('-')[0]
    #         if x == entry:
    #             reg_results[x].append(reg_auc[key])

    # conformal_results = {}

    # for x in range(8,9):
    #     conformal_results[str(x)] = []

    # for key in conformal_auc.keys():
    #     for entry in conformal_results.keys():
    #         x = key.split('-')[0]
    #         if x == entry:
    #             conformal_results[x].append(conformal_auc[key])

    return (
        target_auc,
        imc_results,
        imc_ref_results,
        imc_transition_counts_ref,
        imc_ref_splitting_results,
        imc_transition_counts_ref_splitting,
    )


def plotting(
    coarse,
    target_auc,
    imc_results,
    imc_transition_counts,
    imc_ref_results,
    imc_transition_counts_ref,
    horizon,
    initial_amount,
    imc_ref_splitting_results,
    imc_transition_counts_ref_splitting,
    out_path,
):

    RS_sets = []
    R_sets = []
    NR_sets = []

    # REG_sets = []
    # CONF_sets = []

    for key in imc_ref_splitting_results.keys():
        total_state_count = []
        total = 0
        for val in imc_transition_counts_ref_splitting[key]:
            total += val
            total_state_count.append(total)

        RS_sets.append((total_state_count, imc_ref_splitting_results[key]))

    for key in imc_ref_results.keys():
        total_state_count = []
        total = 0
        for val in imc_transition_counts_ref[key]:
            total += val
            total_state_count.append(total)

        R_sets.append((total_state_count, imc_ref_results[key]))

    for key in imc_results.keys():
        total_state_count = []
        total = 0
        for val in imc_transition_counts[key]:
            total += val
            total_state_count.append(total)

        NR_sets.append((total_state_count, imc_results[key]))

    # for key in reg_results.keys():
    #     total_state_count = []

    #     for val in regression_ys[key]:
    #         total_state_count.append(val*(horizon+initial_amount))

    #     REG_sets.append((total_state_count, reg_results[key]))

    # for key in conformal_results.keys():
    #     total_state_count = []

    #     for val in conformal_ys[key]:
    #         total_state_count.append(val*(horizon+initial_amount))

    #     CONF_sets.append((total_state_count, conformal_results[key]))

    # log = True
    log = False

    # REFINEMENT AVERAGE PERFORMANCE
    transitions_data = []
    auc_data = []

    for entry in R_sets:
        transitions = entry[0]
        auc_daum = entry[1]

        transitions_data.append(transitions)
        auc_data.append(auc_daum)

    # Find common x range for interpolation
    min_x = max(min(transitions) for transitions in transitions_data)
    max_x = min(max(transitions) for transitions in transitions_data)
    x_values = np.linspace(min_x, max_x, 500)

    # Interpolate all runs to common x values
    interpolated_auc = []
    for auc, transitions in zip(auc_data, transitions_data):
        if log:
            auc = np.log10(auc)
        interpolated = np.interp(x_values, transitions, auc)
        if log:
            interpolated = np.power(10, interpolated)
        interpolated_auc.append(interpolated)

    # Calculate mean and std for interpolated y values
    auc_array = np.array(interpolated_auc)
    mean_auc = np.mean(auc_array, axis=0)
    std_auc = np.std(auc_array, axis=0)
    min_auc = np.min(auc_array, axis=0)
    max_auc = np.max(auc_array, axis=0)

    plt.figure()
    fig, ax = plt.subplots(figsize=(20, 10))
    y = np.full_like(x_values, target_auc)
    ax.plot(
        x_values,
        y,
        color="black",
        label="Target Monitor",
        linewidth=5,
        linestyle="-",
        marker="D",
    )

    # Plot mean line
    ax.plot(
        x_values,
        mean_auc,
        color="blue",
        label="Refinement",
        linewidth=5,
        linestyle=":",
    )

    # Add shaded area for spread
    ax.fill_between(
        x_values,
        mean_auc - std_auc,
        mean_auc + std_auc,
        alpha=0.2,
        color="blue",
    )

    # REFINEMENT WITH SPLITTING AVERAGE PERFORMANCE
    transitions_data = []
    auc_data = []

    for entry in RS_sets:
        transitions = entry[0]
        auc_daum = entry[1]

        transitions_data.append(transitions)
        auc_data.append(auc_daum)

    # Find common x range for interpolation
    min_x = max(min(transitions) for transitions in transitions_data)
    max_x = min(max(transitions) for transitions in transitions_data)
    x_values = np.linspace(min_x, max_x, 500)

    # Interpolate all runs to common x values
    interpolated_auc = []
    for auc, transitions in zip(auc_data, transitions_data):
        if log:
            auc = np.log10(auc)
        interpolated = np.interp(x_values, transitions, auc)
        if log:
            interpolated = np.power(10, interpolated)
        interpolated_auc.append(interpolated)

    # Calculate mean and std for interpolated y values
    auc_array = np.array(interpolated_auc)
    mean_auc = np.mean(auc_array, axis=0)
    std_auc = np.std(auc_array, axis=0)
    min_auc = np.min(auc_array, axis=0)
    max_auc = np.max(auc_array, axis=0)

    # Plot mean line
    ax.plot(
        x_values,
        mean_auc,
        color="aqua",
        label="Refinement with splitting",
        linewidth=5,
        linestyle="-.",
    )

    # Add shaded area for spread
    ax.fill_between(
        x_values,
        mean_auc - std_auc,
        mean_auc + std_auc,
        alpha=0.2,
        color="aqua",
    )

    # NO REFINEMENT AVERAGE PERFORMANCE

    N_transitions_data = []
    N_auc_data = []

    for entry in NR_sets:
        transitions = entry[0]
        auc_daum = entry[1]

        N_transitions_data.append(transitions)
        N_auc_data.append(auc_daum)

    # Find common x range for interpolation
    N_min_x = max(min(transitions) for transitions in N_transitions_data)
    N_max_x = min(max(transitions) for transitions in N_transitions_data)
    N_x_values = np.linspace(N_min_x, N_max_x, 500)

    # Interpolate all runs to common x values
    N_interpolated_auc = []
    for auc, transitions in zip(N_auc_data, N_transitions_data):
        if log:
            auc = np.log10(auc)
        N_interpolated = np.interp(N_x_values, transitions, auc)
        if log:
            N_interpolated = np.power(10, N_interpolated)
        N_interpolated_auc.append(N_interpolated)

    # Calculate mean and std for interpolated y values
    N_auc_array = np.array(N_interpolated_auc)
    N_mean_auc = np.mean(N_auc_array, axis=0)
    N_std_auc = np.std(N_auc_array, axis=0)
    N_min_auc = np.min(N_auc_array, axis=0)
    N_max_auc = np.max(N_auc_array, axis=0)

    # Plot mean line
    ax.plot(
        N_x_values,
        N_mean_auc,
        color="red",
        label="No refinement",
        linestyle="--",
        linewidth=5,
    )

    # Add shaded area for spread
    ax.fill_between(
        N_x_values,
        N_mean_auc - N_std_auc,
        N_mean_auc + N_std_auc,
        alpha=0.2,
        color="red",
    )

    
    if coarse:
        plt.savefig(
            f"{out_path}/rq_2_{args.mc}_coarse_AUC_ref_no_ref.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            f"{out_path}/rq_2_{args.mc}_AUC_ref_no_ref.pdf",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def fn_fp_comparison_model_based(
        coarse,
        alarms,imc_risks,
        imc_risks_ref,
        target_risks,
        imc_risks_ref_splitting, 
        out_path):

    
    # iHMM
    imc_final_risks = {}

    ys = []
    for key in imc_risks.keys():
        ys.append(int(key.split("-")[1]))

    for key in imc_risks.keys():
        if key.split("-")[1] == str(max(ys)):
            imc_final_risks[key.split("-")[0]] = imc_risks[key]

    # iHMM ref
    imc_ref_final_risks = {}

    imc_ref_ys = []
    for key in imc_risks_ref.keys():
        imc_ref_ys.append(int(key.split("-")[1]))

    for key in imc_risks_ref.keys():
        if key.split("-")[1] == str(max(imc_ref_ys)):
            imc_ref_final_risks[key.split("-")[0]] = imc_risks_ref[key]


    # iHMM ref with splitting
    imc_ref_splitting_final_risks = {}

    imc_ref_splitting_ys = []
    for key in imc_risks_ref_splitting.keys():
        imc_ref_splitting_ys.append(int(key.split("-")[1]))

    for key in imc_risks_ref_splitting.keys():
        if key.split("-")[1] == str(max(imc_ref_splitting_ys)):
            imc_ref_splitting_final_risks[key.split("-")[0]] = imc_risks_ref_splitting[key]

        
    thresholds = [t / 1000 for t in range(0, 1001)]
    target_risks = np.array(target_risks,  dtype=float)
    alarms = np.array(alarms)

    target_fnr = []
    target_fpr = []

    imc_fnr = {}
    imc_fpr = {}
   
    imc_ref_fnr = {}
    imc_ref_fpr = {}

    imc_ref_splitting_fnr = {}
    imc_ref_splitting_fpr = {}


    for key in imc_final_risks.keys(): 
            imc_fnr[key] = []
            imc_fpr[key] = []

    for key in imc_ref_final_risks.keys(): 
            imc_ref_fnr[key] = []
            imc_ref_fpr[key] = []

    for key in imc_ref_splitting_final_risks.keys(): 
            imc_ref_splitting_fnr[key] = []
            imc_ref_splitting_fpr[key] = []


    for t in thresholds: 

        actual_positives = np.sum(alarms == 1)
        actual_negatives = np.sum(alarms == 0)
        total = len(alarms)

        #Target
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

    
        for key in imc_ref_final_risks.keys(): 
                
                predictions_imc_ref =  np.array(imc_ref_final_risks[key], dtype=float) >= t

                fp_imc_ref = np.sum((predictions_imc_ref == 1) & (alarms == 0))
                fn_imc_ref = np.sum((predictions_imc_ref == 0) & (alarms == 1))
                    
                fnr_imc_ref = fn_imc_ref / actual_positives if actual_positives > 0 else 0.0
                fpr_imc_ref = fp_imc_ref / actual_negatives if actual_negatives > 0 else 0.0

                imc_ref_fnr[key].append(fnr_imc_ref)
                imc_ref_fpr[key].append(fpr_imc_ref)


        for key in imc_ref_splitting_final_risks.keys(): 
                
                predictions_imc_ref_splitting =  np.array(imc_ref_splitting_final_risks[key], dtype=float) >= t

                fp_imc_ref_splitting = np.sum((predictions_imc_ref_splitting == 1) & (alarms == 0))
                fn_imc_ref_splitting = np.sum((predictions_imc_ref_splitting == 0) & (alarms == 1))
                    
                fnr_imc_ref_splitting = fn_imc_ref_splitting / actual_positives if actual_positives > 0 else 0.0
                fpr_imc_ref_splitting = fp_imc_ref_splitting / actual_negatives if actual_negatives > 0 else 0.0

                imc_ref_splitting_fnr[key].append(fnr_imc_ref_splitting)
                imc_ref_splitting_fpr[key].append(fpr_imc_ref_splitting)


    fig, ax = plt.subplots(figsize=(8, 5)) 

    auc_fnr_target = np.trapz(target_fnr, thresholds)
    auc_fpr_target = np.trapz(target_fpr, thresholds)

    ax.plot(thresholds, target_fnr, label=f'Target FNR: (AUC {auc_fnr_target:.3f})', color='black', linestyle= ':')
    ax.plot(thresholds, target_fpr , label=f'Target FPR: (AUC {auc_fpr_target:.3f})', color='black', linestyle= '--')

    #No Refinement
    
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

    ax.plot(thresholds, imc_fnr_mean, label=f'No refinement mean FNR: (AUC {imc_fnr_auc:.3f})', color='red', linestyle= ':')

    plt.fill_between(
    thresholds,
    imc_fnr_mean - imc_fnr_std,   
    imc_fnr_mean + imc_fnr_std,   
    color="red",
    alpha=0.2
    )

    imc_FPRs = np.array(imc_FPRs)

    imc_fpr_mean = np.mean(imc_FPRs, axis=0)
    imc_fpr_std = np.std(imc_FPRs, axis=0)

    ax.plot(thresholds, imc_fpr_mean, label=f'No refinement mean FPR: (AUC {imc_fpr_auc:.3f})', color='red', linestyle= '--')

    plt.fill_between(
    thresholds,
    imc_fpr_mean - imc_fpr_std,   
    imc_fpr_mean + imc_fpr_std,   
    color="red",
    alpha=0.2
    )

    #Refinement


    imc_ref_FNRs = []
    imc_ref_FPRs = []

    imc_ref_fnr_aucs = []
    imc_ref_fpr_aucs = []


    for key in imc_ref_final_risks.keys():
        imc_ref_FNRs.append(imc_ref_fnr[key]) 
        imc_ref_FPRs.append(imc_ref_fpr[key])
        imc_ref_fnr_aucs.append(np.trapz(imc_ref_fnr[key], thresholds))
        imc_ref_fpr_aucs.append(np.trapz(imc_ref_fpr[key], thresholds))

    imc_ref_FNRs = np.array(imc_ref_FNRs)    

    imc_ref_fnr_mean = np.mean(imc_ref_FNRs, axis=0)
    imc_ref_fnr_std = np.std(imc_ref_FNRs, axis=0)

    imc_ref_fnr_aucs = np.array(imc_ref_fnr_aucs)
    imc_ref_fnr_auc = np.mean(imc_ref_fnr_aucs, axis=0)

    imc_ref_fpr_aucs = np.array(imc_ref_fpr_aucs)
    imc_ref_fpr_auc = np.mean(imc_ref_fpr_aucs, axis=0)


    ax.plot(thresholds, imc_ref_fnr_mean, label=f'Refinement mean FNR: (AUC {imc_ref_fnr_auc:.3f})', color='blue', linestyle= ':')

    plt.fill_between(
    thresholds,
    imc_ref_fnr_mean - imc_ref_fnr_std,   
    imc_ref_fnr_mean + imc_ref_fnr_std,   
    color="blue",
    alpha=0.2
    )

    imc_ref_FPRs = np.array(imc_ref_FPRs)

    imc_ref_fpr_mean = np.mean(imc_ref_FPRs, axis=0)
    imc_ref_fpr_std = np.std(imc_ref_FPRs, axis=0)

    ax.plot(thresholds, imc_ref_fpr_mean, label=f'Refinement mean FPR: (AUC {imc_ref_fpr_auc:.3f})', color='blue', linestyle= '--')

    plt.fill_between(
    thresholds,
    imc_ref_fpr_mean - imc_ref_fpr_std,   
    imc_ref_fpr_mean + imc_ref_fpr_std,   
    color="blue",
    alpha=0.2
    )

    #Refinement with splitting

    imc_ref_splitting_FNRs = []
    imc_ref_splitting_FPRs = []

    imc_ref_splitting_fnr_aucs = []
    imc_ref_splitting_fpr_aucs = []


    for key in imc_ref_splitting_final_risks.keys():
        imc_ref_splitting_FNRs.append(imc_ref_splitting_fnr[key]) 
        imc_ref_splitting_FPRs.append(imc_ref_splitting_fpr[key])
        imc_ref_splitting_fnr_aucs.append(np.trapz(imc_ref_splitting_fnr[key], thresholds))
        imc_ref_splitting_fpr_aucs.append(np.trapz(imc_ref_splitting_fpr[key], thresholds))

    imc_ref_splitting_FNRs = np.array(imc_ref_splitting_FNRs)    

    imc_ref_splitting_fnr_mean = np.mean(imc_ref_splitting_FNRs, axis=0)
    imc_ref_splitting_fnr_std = np.std(imc_ref_splitting_FNRs, axis=0)

    imc_ref_splitting_fnr_aucs = np.array(imc_ref_splitting_fnr_aucs)
    imc_ref_splitting_fnr_auc = np.mean(imc_ref_splitting_fnr_aucs, axis=0)

    imc_ref_splitting_fpr_aucs = np.array(imc_ref_splitting_fpr_aucs)
    imc_ref_splitting_fpr_auc = np.mean(imc_ref_splitting_fpr_aucs, axis=0)


    ax.plot(thresholds, imc_ref_splitting_fnr_mean, label=f'Refinement with Splitting mean FNR: (AUC {imc_ref_splitting_fnr_auc:.3f})', color='aqua', linestyle= ':')

    plt.fill_between(
    thresholds,
    imc_ref_splitting_fnr_mean - imc_ref_splitting_fnr_std,   
    imc_ref_splitting_fnr_mean + imc_ref_splitting_fnr_std,   
    color="aqua",
    alpha=0.2
    )

    imc_ref_splitting_FPRs = np.array(imc_ref_splitting_FPRs)

    imc_ref_splitting_fpr_mean = np.mean(imc_ref_splitting_FPRs, axis=0)
    imc_ref_splitting_fpr_std = np.std(imc_ref_splitting_FPRs, axis=0)

    ax.plot(thresholds, imc_ref_splitting_fpr_mean, label=f'Refinement with Splitting mean FPR: (AUC {imc_ref_splitting_fpr_auc:.3f})', color='aqua', linestyle= '--')

    plt.fill_between(
    thresholds,
    imc_ref_splitting_fpr_mean - imc_ref_splitting_fpr_std,   
    imc_ref_splitting_fpr_mean + imc_ref_splitting_fpr_std,   
    color="aqua",
    alpha=0.2
    )


    ax.set_xlabel('Threshold')
    ax.set_ylabel('Rate')
    ax.legend(loc="right", fontsize=5)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


    if coarse:
        fig.savefig(
                f"{out_path}/rq_2_{args.mc}_coarse_FN_FP_model_based.pdf",
                dpi=300,
                bbox_inches="tight",
            )
    else:
        fig.savefig(
                f"{out_path}/rq_2_{args.mc}_FN_FP_model_based.pdf",
                dpi=300,
                bbox_inches="tight",
            )

    plt.show()

def fn_fp_AUC_model_based(
        coarse,
        alarms,imc_risks,
        imc_risks_ref,
        target_risks,
        imc_risks_ref_splitting, 
        imc_transition_counts,
        imc_transition_counts_ref,
        imc_transition_counts_ref_splitting,
        out_path):

    
    thresholds = [t / 1000 for t in range(0, 1001)]
    target_risks = np.array(target_risks,  dtype=float)
    alarms = np.array(alarms)

    target_fnr = []
    target_fpr = []

    imc_fnr = {}
    imc_fpr = {}
   
    imc_ref_fnr = {}
    imc_ref_fpr = {}

    imc_ref_splitting_fnr = {}
    imc_ref_splitting_fpr = {}

    for key in imc_risks.keys(): 
            imc_fnr[key] = []
            imc_fpr[key] = []

    for key in imc_risks_ref.keys(): 
            imc_ref_fnr[key] = []
            imc_ref_fpr[key] = []

    for key in imc_risks_ref_splitting.keys(): 
            imc_ref_splitting_fnr[key] = []
            imc_ref_splitting_fpr[key] = []


    for t in thresholds: 

        actual_positives = np.sum(alarms == 1)
        actual_negatives = np.sum(alarms == 0)
        total = len(alarms)

        #Target
        predictions_target = target_risks >= t

        fp_target = np.sum((predictions_target == 1) & (alarms == 0))
        fn_target = np.sum((predictions_target == 0) & (alarms == 1))
    
        fnr_target = fn_target / actual_positives if actual_positives > 0 else 0.0
        fpr_target = fp_target / actual_negatives if actual_negatives > 0 else 0.0
        
        target_fnr.append(fnr_target)
        target_fpr.append(fpr_target)

    
        for key in imc_risks.keys(): 

                predictions_imc = np.array(imc_risks[key], dtype=float) >= t

                fp_imc = np.sum((predictions_imc == True) & (alarms == 0))
                fn_imc = np.sum((predictions_imc == False) & (alarms == 1))
                    
                fnr_imc = fn_imc / actual_positives if actual_positives > 0 else 0.0
                fpr_imc = fp_imc / actual_negatives if actual_negatives > 0 else 0.0

                imc_fnr[key].append(fnr_imc)
                imc_fpr[key].append(fpr_imc) 

    
        for key in imc_risks_ref.keys(): 
                
                predictions_imc_ref =  np.array(imc_risks_ref[key], dtype=float) >= t

                fp_imc_ref = np.sum((predictions_imc_ref == 1) & (alarms == 0))
                fn_imc_ref = np.sum((predictions_imc_ref == 0) & (alarms == 1))
                    
                fnr_imc_ref = fn_imc_ref / actual_positives if actual_positives > 0 else 0.0
                fpr_imc_ref = fp_imc_ref / actual_negatives if actual_negatives > 0 else 0.0

                imc_ref_fnr[key].append(fnr_imc_ref)
                imc_ref_fpr[key].append(fpr_imc_ref)


        for key in imc_risks_ref_splitting.keys(): 
                
                predictions_imc_ref_splitting =  np.array(imc_risks_ref_splitting[key], dtype=float) >= t

                fp_imc_ref_splitting = np.sum((predictions_imc_ref_splitting == 1) & (alarms == 0))
                fn_imc_ref_splitting = np.sum((predictions_imc_ref_splitting == 0) & (alarms == 1))
                    
                fnr_imc_ref_splitting = fn_imc_ref_splitting / actual_positives if actual_positives > 0 else 0.0
                fpr_imc_ref_splitting = fp_imc_ref_splitting / actual_negatives if actual_negatives > 0 else 0.0

                imc_ref_splitting_fnr[key].append(fnr_imc_ref_splitting)
                imc_ref_splitting_fpr[key].append(fpr_imc_ref_splitting)


    fig, ax = plt.subplots(figsize=(8, 5)) 

    auc_fnr_target = np.trapz(target_fnr, thresholds)
    auc_fpr_target = np.trapz(target_fpr, thresholds)

    imc_transition_counts_accumulated = {}
    imc_transition_counts_ref_accumulated = {}
    imc_transition_counts_ref_splitting_accumulated = {}

    for x in imc_transition_counts.keys():
        imc_transition_counts_accumulated[x] = list(accumulate(imc_transition_counts[x]))

    for y in imc_transition_counts_ref.keys():
        imc_transition_counts_ref_accumulated[y] = list(accumulate(imc_transition_counts_ref[y]))
    
    for x in imc_transition_counts_ref_splitting.keys():
        imc_transition_counts_ref_splitting_accumulated[x] = list(accumulate(imc_transition_counts_ref_splitting[x]))

    y_range = 0 

    for x in imc_transition_counts.keys(): 
        if max([max(imc_transition_counts_accumulated[x]),max(imc_transition_counts_ref_accumulated[x]),max(imc_transition_counts_ref_splitting_accumulated[x])]) > y_range: 
            y_range = max([max(imc_transition_counts_accumulated[x]),max(imc_transition_counts_ref_accumulated[x]),max(imc_transition_counts_ref_splitting_accumulated[x])])

    ax.plot([0, int(y_range)], [auc_fnr_target, auc_fnr_target], label = 'Target FNR AUC', color = 'black', linestyle= '--')
    ax.plot([0, int(y_range)], [auc_fpr_target, auc_fpr_target], label = 'Target FPR AUC', color = 'black', linestyle= ':') 

    #No Refinement
    
    imc_fnr_aucs = {}
    imc_fpr_aucs = {}
    

    for key in imc_risks.keys():
        imc_fnr_aucs[key] = np.trapz(imc_fnr[key], thresholds)
        imc_fpr_aucs[key] = np.trapz(imc_fpr[key], thresholds)

    #fnr
    imc_fnr_aucs_sorted = {}

    for key in imc_fnr_aucs.keys(): 
        imc_fnr_aucs_sorted[key.split("-")[0]] = []

    for x in imc_fnr_aucs.keys():
        for y in imc_fnr_aucs_sorted.keys(): 
            if x.split("-")[0] == y: 
                imc_fnr_aucs_sorted[y].append([imc_fnr_aucs[x]])

    #fpr
    imc_fpr_aucs_sorted = {}

    for key in imc_fpr_aucs.keys(): 
        imc_fpr_aucs_sorted[key.split("-")[0]] = []

    for x in imc_fpr_aucs.keys():
        for y in imc_fpr_aucs_sorted.keys(): 
            if x.split("-")[0] == y: 
                imc_fpr_aucs_sorted[y].append([imc_fpr_aucs[x]])



    no_ref_x = []
    fnr_no_rf_y = []
    fpr_no_rf_y = []

    for x in imc_fnr_aucs_sorted.keys():
        no_ref_x.append(imc_transition_counts_accumulated[x])
        fnr_no_rf_y.append(imc_fnr_aucs_sorted[x])
        fpr_no_rf_y.append(imc_fpr_aucs_sorted[x])


    no_ref_x = np.array(no_ref_x)
    mean_no_ref_x = np.mean(no_ref_x, axis=0)

    fnr_no_rf_y = np.array(fnr_no_rf_y)
    mean_fnr_no_rf_y = np.mean(fnr_no_rf_y, axis=0)
    std_fnr_no_rf_y = np.std(fnr_no_rf_y, axis=0)

    fpr_no_rf_y = np.array(fpr_no_rf_y)
    mean_fpr_no_rf_y = np.mean(fpr_no_rf_y, axis=0)
    std_fpr_no_rf_y = np.std(fpr_no_rf_y, axis=0)

    ax.plot(mean_no_ref_x, mean_fnr_no_rf_y, label = 'No refinement FNR AUC', color='red', linestyle= '--')
    ax.plot(mean_no_ref_x, mean_fpr_no_rf_y, label = 'No refinement FPR AUC', color='red', linestyle= ':')

    plt.fill_between(
    mean_no_ref_x.flatten(),
    (mean_fnr_no_rf_y - std_fnr_no_rf_y).flatten(),   
    (mean_fnr_no_rf_y + std_fnr_no_rf_y).flatten(),   
    color="red",
    alpha=0.2
    )

    plt.fill_between(
    mean_no_ref_x.flatten(),
    (mean_fpr_no_rf_y - std_fpr_no_rf_y).flatten(),   
    (mean_fpr_no_rf_y + std_fpr_no_rf_y).flatten(),   
    color="red",
    alpha=0.2
    )


    #Refinement
    
    imc_ref_fnr_aucs = {}
    imc_ref_fpr_aucs = {}

    for key in imc_risks_ref.keys():
        imc_ref_fnr_aucs[key] = np.trapz(imc_ref_fnr[key], thresholds)
        imc_ref_fpr_aucs[key] = np.trapz(imc_ref_fpr[key], thresholds)


    #fnr
    imc_ref_fnr_aucs_sorted = {}

    for key in imc_ref_fnr_aucs.keys(): 
        imc_ref_fnr_aucs_sorted[key.split("-")[0]] = []


    for x in imc_ref_fnr_aucs.keys():
        for y in imc_ref_fnr_aucs_sorted.keys(): 
            if x.split("-")[0] == y: 
                imc_ref_fnr_aucs_sorted[y].append([imc_ref_fnr_aucs[x]])


    #fpr
    imc_ref_fpr_aucs_sorted = {}

    for key in imc_ref_fpr_aucs.keys(): 
        imc_ref_fpr_aucs_sorted[key.split("-")[0]] = []


    for x in imc_ref_fpr_aucs.keys():
        for y in imc_ref_fpr_aucs_sorted.keys(): 
            if x.split("-")[0] == y: 
                imc_ref_fpr_aucs_sorted[y].append([imc_ref_fpr_aucs[x]])


    ref_x = []
    fnr_rf_y = []
    fpr_rf_y = []


    for x in imc_ref_fpr_aucs_sorted.keys(): 
        ref_x.append(imc_transition_counts_ref_accumulated[x])
        fnr_rf_y.append(imc_ref_fnr_aucs_sorted[x])
        fpr_rf_y.append(imc_ref_fpr_aucs_sorted[x])

    ref_x = np.array(ref_x)
    mean_ref_x = np.mean(ref_x, axis=0)

    fnr_rf_y = np.array(fnr_rf_y)
    mean_fnr_rf_y = np.mean(fnr_rf_y, axis=0)
    std_fnr_rf_y = np.std(fnr_rf_y, axis=0)

    fpr_rf_y = np.array(fpr_rf_y)
    mean_fpr_rf_y = np.mean(fpr_rf_y, axis=0)
    std_fpr_rf_y = np.std(fpr_rf_y, axis=0)

    ax.plot(mean_ref_x, mean_fnr_rf_y, label = 'Refinement FNR AUC', color='blue', linestyle= '--')
    ax.plot(mean_ref_x, mean_fpr_rf_y, label = 'Refinement FPR AUC', color='blue', linestyle= ':')

    plt.fill_between(
    mean_ref_x.flatten(),
    (mean_fnr_rf_y - std_fnr_rf_y).flatten(),   
    (mean_fnr_rf_y + std_fnr_rf_y).flatten(),   
    color="blue",
    alpha=0.2
    )

    plt.fill_between(
    mean_ref_x.flatten(),
    (mean_fpr_rf_y - std_fpr_rf_y).flatten(),   
    (mean_fpr_rf_y + std_fpr_rf_y).flatten(),   
    color="blue",
    alpha=0.2
    )

    #Refinement with Splitting
    
    imc_ref_splitting_fnr_aucs = {}
    imc_ref_splitting_fpr_aucs = {}

    for key in imc_risks_ref_splitting.keys():
        imc_ref_splitting_fnr_aucs[key] = np.trapz(imc_ref_splitting_fnr[key], thresholds)
        imc_ref_splitting_fpr_aucs[key] = np.trapz(imc_ref_splitting_fpr[key], thresholds)


    #fnr
    imc_ref_splitting_fnr_aucs_sorted = {}

    for key in imc_ref_splitting_fnr_aucs.keys(): 
        imc_ref_splitting_fnr_aucs_sorted[key.split("-")[0]] = []


    for x in imc_ref_splitting_fnr_aucs.keys():
        for y in imc_ref_splitting_fnr_aucs_sorted.keys(): 
            if x.split("-")[0] == y: 
                imc_ref_splitting_fnr_aucs_sorted[y].append([imc_ref_splitting_fnr_aucs[x]])


    #fpr
    imc_ref_splitting_fpr_aucs_sorted = {}

    for key in imc_ref_splitting_fpr_aucs.keys(): 
        imc_ref_splitting_fpr_aucs_sorted[key.split("-")[0]] = []


    for x in imc_ref_splitting_fpr_aucs.keys():
        for y in imc_ref_splitting_fpr_aucs_sorted.keys(): 
            if x.split("-")[0] == y: 
                imc_ref_splitting_fpr_aucs_sorted[y].append([imc_ref_splitting_fpr_aucs[x]])

    ref_split_x = []
    fnr_rf_split_y = []
    fpr_rf_split_y = []


    for x in imc_ref_splitting_fpr_aucs_sorted.keys(): 
        ref_split_x.append(imc_transition_counts_ref_splitting_accumulated[x])
        fnr_rf_split_y.append(imc_ref_splitting_fnr_aucs_sorted[x])
        fpr_rf_split_y.append(imc_ref_splitting_fpr_aucs_sorted[x])

    
    ref_split_x = np.array(ref_split_x)
    mean_ref_split_x = np.mean(ref_split_x, axis=0)

    fnr_rf_split_y = np.array(fnr_rf_split_y)
    mean_fnr_rf_split_y = np.mean(fnr_rf_split_y, axis=0)
    std_fnr_rf_split_y = np.std(fnr_rf_split_y, axis=0)

    fpr_rf_split_y = np.array(fpr_rf_split_y)
    mean_fpr_rf_split_y = np.mean(fpr_rf_split_y, axis=0)
    std_fpr_rf_split_y = np.std(fpr_rf_split_y, axis=0)

    ax.plot(mean_ref_split_x, mean_fnr_rf_split_y, label = 'Refinement with Splitting FNR AUC', color='aqua', linestyle= '--')
    ax.plot(mean_ref_split_x, mean_fpr_rf_split_y, label = 'Refinement with Splitting FPR AUC', color='aqua', linestyle= ':')

    plt.fill_between(
    mean_ref_split_x.flatten(),
    (mean_fnr_rf_split_y - std_fnr_rf_split_y).flatten(),   
    (mean_fnr_rf_split_y + std_fnr_rf_split_y).flatten(),   
    color="aqua",
    alpha=0.2
    )

    plt.fill_between(
    mean_ref_split_x.flatten(),
    (mean_fpr_rf_split_y - std_fpr_rf_split_y).flatten(),   
    (mean_fpr_rf_split_y + std_fpr_rf_split_y).flatten(),   
    color="aqua",
    alpha=0.2
    )

    ax.set_ylabel('AUC')
    ax.set_xlabel('Explored States')
    ax.legend(loc="right", fontsize=6)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


    if coarse:
        fig.savefig(
                f"{out_path}/rq_2_{args.mc}_coarse_FN_FP_AUC_model_based.pdf",
                dpi=300,
                bbox_inches="tight",
            )
    else:
        fig.savefig(
                f"{out_path}/rq_2_{args.mc}_FN_FP_AUC_model_based.pdf",
                dpi=300,
                bbox_inches="tight",
            )

    plt.show()



def fn_fp_comparison_model_based_vs_model_free(
        high_st,
        coarse,
        alarms,
        imc_risks_ref,
        imc_risks_ref_splitting,
        regression_risks,
        conformal_risks,
        target_risks,
        out_path):

    
    # iHMM ref
    imc_ref_final_risks = {}

    imc_ref_ys = []
    for key in imc_risks_ref.keys():
        imc_ref_ys.append(int(key.split("-")[1]))

    for key in imc_risks_ref.keys():
        if key.split("-")[1] == str(max(imc_ref_ys)):
            imc_ref_final_risks[key.split("-")[0]] = imc_risks_ref[key]


    # iHMM ref with splitting
    imc_ref_splitting_final_risks = {}

    imc_ref_splitting_ys = []
    for key in imc_risks_ref_splitting.keys():
        imc_ref_splitting_ys.append(int(key.split("-")[1]))

    for key in imc_risks_ref_splitting.keys():
        if key.split("-")[1] == str(max(imc_ref_splitting_ys)):
            imc_ref_splitting_final_risks[key.split("-")[0]] = imc_risks_ref_splitting[key]

    #Regression 
    reg_final_risks = {}
    for key in regression_risks.keys():
        reg_final_risks[key] = regression_risks[key]

    # Confromal Prediction
    conformal_final_risks = {}

    for key in conformal_risks.keys():
        conformal_final_risks[key] = conformal_risks[key]

     
    thresholds = [t / 1000 for t in range(0, 1001)]
    target_risks = np.array(target_risks,  dtype=float)
    alarms = np.array(alarms)

    target_fnr = []
    target_fpr = []

    imc_ref_fnr = {}
    imc_ref_fpr = {}

    imc_ref_splitting_fnr = {}
    imc_ref_splitting_fpr = {}

    reg_fnr = {}
    reg_fpr = {}

    conformal_fnr = {}
    conformal_fpr = {}


    for key in imc_ref_final_risks.keys(): 
            imc_ref_fnr[key] = []
            imc_ref_fpr[key] = []

    for key in imc_ref_splitting_final_risks.keys(): 
            imc_ref_splitting_fnr[key] = []
            imc_ref_splitting_fpr[key] = []

    for key in reg_final_risks.keys(): 
            reg_fnr[key] = []
            reg_fpr[key] = []

    for key in conformal_final_risks.keys(): 
            conformal_fnr[key] = []
            conformal_fpr[key] = []



    for t in thresholds: 

        actual_positives = np.sum(alarms == 1)
        actual_negatives = np.sum(alarms == 0)
        total = len(alarms)

        #Target
        predictions_target = target_risks >= t

        fp_target = np.sum((predictions_target == 1) & (alarms == 0))
        fn_target = np.sum((predictions_target == 0) & (alarms == 1))
    
        fnr_target = fn_target / actual_positives if actual_positives > 0 else 0.0
        fpr_target = fp_target / actual_negatives if actual_negatives > 0 else 0.0
        
        target_fnr.append(fnr_target)
        target_fpr.append(fpr_target)
    
        #Refinement

        for key in imc_ref_final_risks.keys(): 
                
                predictions_imc_ref =  np.array(imc_ref_final_risks[key], dtype=float) >= t

                fp_imc_ref = np.sum((predictions_imc_ref == 1) & (alarms == 0))
                fn_imc_ref = np.sum((predictions_imc_ref == 0) & (alarms == 1))
                    
                fnr_imc_ref = fn_imc_ref / actual_positives if actual_positives > 0 else 0.0
                fpr_imc_ref = fp_imc_ref / actual_negatives if actual_negatives > 0 else 0.0

                imc_ref_fnr[key].append(fnr_imc_ref)
                imc_ref_fpr[key].append(fpr_imc_ref)

        #Refinement with splitting

        for key in imc_ref_splitting_final_risks.keys(): 
                
                predictions_imc_ref_splitting =  np.array(imc_ref_splitting_final_risks[key], dtype=float) >= t

                fp_imc_ref_splitting = np.sum((predictions_imc_ref_splitting == 1) & (alarms == 0))
                fn_imc_ref_splitting = np.sum((predictions_imc_ref_splitting == 0) & (alarms == 1))
                    
                fnr_imc_ref_splitting = fn_imc_ref_splitting / actual_positives if actual_positives > 0 else 0.0
                fpr_imc_ref_splitting = fp_imc_ref_splitting / actual_negatives if actual_negatives > 0 else 0.0

                imc_ref_splitting_fnr[key].append(fnr_imc_ref_splitting)
                imc_ref_splitting_fpr[key].append(fpr_imc_ref_splitting)

        #Regression 

        for key in reg_final_risks.keys(): 
                
                predictions_reg =  np.array(reg_final_risks[key], dtype=float) >= t

                fp_reg = np.sum((predictions_reg == 1) & (alarms == 0))
                fn_reg = np.sum((predictions_reg == 0) & (alarms == 1))
                    
                fnr_reg = fn_reg/ actual_positives if actual_positives > 0 else 0.0
                fpr_reg = fp_reg / actual_negatives if actual_negatives > 0 else 0.0

                reg_fnr[key].append(fnr_reg)
                reg_fpr[key].append(fpr_reg)

        #Conformal Prediction

        for key in conformal_final_risks.keys(): 
                
                predictions_conformal =  np.array(conformal_final_risks[key], dtype=float) >= t

                fp_conformal = np.sum((predictions_conformal == 1) & (alarms == 0))
                fn_conformal = np.sum((predictions_conformal == 0) & (alarms == 1))
                    
                fnr_conformal = fn_conformal/ actual_positives if actual_positives > 0 else 0.0
                fpr_conformal = fp_conformal / actual_negatives if actual_negatives > 0 else 0.0

                conformal_fnr[key].append(fnr_conformal)
                conformal_fpr[key].append(fpr_conformal)


    fig, ax = plt.subplots(figsize=(8, 5)) 

    auc_fnr_target = np.trapz(target_fnr, thresholds)
    auc_fpr_target = np.trapz(target_fpr, thresholds)


    ax.plot(thresholds, target_fnr, label=f'Target FNR: (AUC {auc_fnr_target:.3f})', color='black', linestyle= ':')
    ax.plot(thresholds, target_fpr, label=f'Target FPR: (AUC {auc_fpr_target:.3f})', color='black', linestyle= '--')

    #Refinement

    imc_ref_FNRs = []
    imc_ref_FPRs = []

    imc_ref_fnr_aucs = []
    imc_ref_fpr_aucs = []


    for key in imc_ref_final_risks.keys():
        imc_ref_FNRs.append(imc_ref_fnr[key]) 
        imc_ref_FPRs.append(imc_ref_fpr[key])
        imc_ref_fnr_aucs.append(np.trapz(imc_ref_fnr[key], thresholds))
        imc_ref_fpr_aucs.append(np.trapz(imc_ref_fpr[key], thresholds))

    imc_ref_FNRs = np.array(imc_ref_FNRs)    

    imc_ref_fnr_mean = np.mean(imc_ref_FNRs, axis=0)
    imc_ref_fnr_std = np.std(imc_ref_FNRs, axis=0)

    imc_ref_fnr_aucs = np.array(imc_ref_fnr_aucs)
    imc_ref_fnr_auc = np.mean(imc_ref_fnr_aucs, axis=0)

    imc_ref_fpr_aucs = np.array(imc_ref_fpr_aucs)
    imc_ref_fpr_auc = np.mean(imc_ref_fpr_aucs, axis=0)


    ax.plot(thresholds, imc_ref_fnr_mean, label=f'Refinement mean FNR: (AUC {imc_ref_fnr_auc:.3f})', color='blue', linestyle= ':')

    plt.fill_between(
    thresholds,
    imc_ref_fnr_mean - imc_ref_fnr_std,   
    imc_ref_fnr_mean + imc_ref_fnr_std,   
    color="blue",
    alpha=0.2
    )

    imc_ref_FPRs = np.array(imc_ref_FPRs)

    imc_ref_fpr_mean = np.mean(imc_ref_FPRs, axis=0)
    imc_ref_fpr_std = np.std(imc_ref_FPRs, axis=0)

    ax.plot(thresholds, imc_ref_fpr_mean, label=f'Refinement mean FPR: (AUC {imc_ref_fpr_auc:.3f})', color='blue', linestyle= '--')

    plt.fill_between(
    thresholds,
    imc_ref_fpr_mean - imc_ref_fpr_std,   
    imc_ref_fpr_mean + imc_ref_fpr_std,   
    color="blue",
    alpha=0.2
    )

    #Refinement with splitting

    imc_ref_splitting_FNRs = []
    imc_ref_splitting_FPRs = []

    imc_ref_splitting_fnr_aucs = []
    imc_ref_splitting_fpr_aucs = []


    for key in imc_ref_splitting_final_risks.keys():
        imc_ref_splitting_FNRs.append(imc_ref_splitting_fnr[key]) 
        imc_ref_splitting_FPRs.append(imc_ref_splitting_fpr[key])
        imc_ref_splitting_fnr_aucs.append(np.trapz(imc_ref_splitting_fnr[key], thresholds))
        imc_ref_splitting_fpr_aucs.append(np.trapz(imc_ref_splitting_fpr[key], thresholds))

    imc_ref_splitting_FNRs = np.array(imc_ref_splitting_FNRs)    

    imc_ref_splitting_fnr_mean = np.mean(imc_ref_splitting_FNRs, axis=0)
    imc_ref_splitting_fnr_std = np.std(imc_ref_splitting_FNRs, axis=0)

    imc_ref_splitting_fnr_aucs = np.array(imc_ref_splitting_fnr_aucs)
    imc_ref_splitting_fnr_auc = np.mean(imc_ref_splitting_fnr_aucs, axis=0)

    imc_ref_splitting_fpr_aucs = np.array(imc_ref_splitting_fpr_aucs)
    imc_ref_splitting_fpr_auc = np.mean(imc_ref_splitting_fpr_aucs, axis=0)


    ax.plot(thresholds, imc_ref_splitting_fnr_mean, label=f'Refinement with Splitting mean FNR: (AUC {imc_ref_splitting_fnr_auc:.3f})', color='aqua', linestyle= ':')

    plt.fill_between(
    thresholds,
    imc_ref_splitting_fnr_mean - imc_ref_splitting_fnr_std,   
    imc_ref_splitting_fnr_mean + imc_ref_splitting_fnr_std,   
    color="aqua",
    alpha=0.2
    )

    imc_ref_splitting_FPRs = np.array(imc_ref_splitting_FPRs)

    imc_ref_splitting_fpr_mean = np.mean(imc_ref_splitting_FPRs, axis=0)
    imc_ref_splitting_fpr_std = np.std(imc_ref_splitting_FPRs, axis=0)

    ax.plot(thresholds, imc_ref_splitting_fpr_mean, label=f'Refinement with Splitting mean FPR: (AUC {imc_ref_splitting_fpr_auc:.3f})', color='aqua', linestyle= '--')

    plt.fill_between(
    thresholds,
    imc_ref_splitting_fpr_mean - imc_ref_splitting_fpr_std,   
    imc_ref_splitting_fpr_mean + imc_ref_splitting_fpr_std,   
    color="aqua",
    alpha=0.2
    )

    #Regression 

    reg_FNRs = []
    reg_FPRs = []

    reg_fnr_aucs = []
    reg_fpr_aucs = []


    for key in reg_final_risks.keys():
        reg_FNRs.append(reg_fnr[key]) 
        reg_FPRs.append(reg_fpr[key])
        reg_fnr_aucs.append(np.trapz(reg_fnr[key], thresholds))
        reg_fpr_aucs.append(np.trapz(reg_fpr[key], thresholds))

    reg_FNRs = np.array(reg_FNRs)    

    reg_fnr_mean = np.mean(reg_FNRs, axis=0)
    reg_fnr_std = np.std(reg_FNRs, axis=0)

    reg_fnr_aucs = np.array(reg_fnr_aucs)
    reg_fnr_auc = np.mean(reg_fnr_aucs, axis=0)

    reg_fpr_aucs = np.array(reg_fpr_aucs)
    reg_fpr_auc = np.mean(reg_fpr_aucs, axis=0)


    ax.plot(thresholds, reg_fnr_mean, label=f'Regression mean FNR: (AUC {reg_fnr_auc:.3f})', color='pink', linestyle= ':')

    plt.fill_between(
    thresholds,
    reg_fnr_mean - reg_fnr_std,   
    reg_fnr_mean + reg_fnr_std,   
    color="pink",
    alpha=0.2
    )

    reg_FPRs = np.array(reg_FPRs)

    reg_fpr_mean = np.mean(reg_FPRs, axis=0)
    reg_fpr_std = np.std(reg_FPRs, axis=0)

    ax.plot(thresholds, reg_fpr_mean, label=f'Regression mean FPR: (AUC {reg_fpr_auc:.3f})', color='pink', linestyle= '--')

    plt.fill_between(
    thresholds,
    reg_fpr_mean - reg_fpr_std,   
    reg_fpr_mean + reg_fpr_std,   
    color="pink",
    alpha=0.2
    )

    #Conformal Prediction

    conformal_FNRs = []
    conformal_FPRs = []

    conformal_fnr_aucs = []
    conformal_fpr_aucs = []


    for key in conformal_final_risks.keys():
        conformal_FNRs.append(conformal_fnr[key]) 
        conformal_FPRs.append(conformal_fpr[key])
        conformal_fnr_aucs.append(np.trapz(conformal_fnr[key], thresholds))
        conformal_fpr_aucs.append(np.trapz(conformal_fpr[key], thresholds))

    conformal_FNRs = np.array(conformal_FNRs)    

    conformal_fnr_mean = np.mean(conformal_FNRs, axis=0)
    conformal_fnr_std = np.std(conformal_FNRs, axis=0)

    conformal_fnr_aucs = np.array(conformal_fnr_aucs)
    conformal_fnr_auc = np.mean(conformal_fnr_aucs, axis=0)

    conformal_fpr_aucs = np.array(conformal_fpr_aucs)
    conformal_fpr_auc = np.mean(conformal_fpr_aucs, axis=0)


    ax.plot(thresholds, conformal_fnr_mean, label=f'Conformal prediction mean FNR: (AUC {conformal_fnr_auc:.3f})', color='orange', linestyle= ':')

    plt.fill_between(
    thresholds,
    conformal_fnr_mean - conformal_fnr_std,   
    conformal_fnr_mean + conformal_fnr_std,   
    color="orange",
    alpha=0.2
    )

    conformal_FPRs = np.array(conformal_FPRs)

    conformal_fpr_mean = np.mean(conformal_FPRs, axis=0)
    conformal_fpr_std = np.std(conformal_FPRs, axis=0)

    ax.plot(thresholds, conformal_fpr_mean, label=f'Conformal prediction mean FPR: (AUC {conformal_fpr_auc:.3f})', color='orange', linestyle= '--')

    plt.fill_between(
    thresholds,
    conformal_fpr_mean - conformal_fpr_std,   
    conformal_fpr_mean + conformal_fpr_std,   
    color="orange",
    alpha=0.2
    )

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Rate')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


    if coarse:
        fig.savefig(
                f"{out_path}/rq_3_{args.mc}_coarse_FN_FP_model_based_vs_model_free.pdf",
                dpi=300,
                bbox_inches="tight",
            )
    else:
        fig.savefig(
                f"{out_path}/rq_3_{args.mc}_FN_FP_model_based_vs_model_free.pdf",
                dpi=300,
                bbox_inches="tight",
            )

    plt.show()


def main_imc(args: argparse.Namespace):
    setup_logging("rq3:" + args.mc + str(args.coarse) + str(args.high_st))

    os.makedirs(args.out, exist_ok=True)

    suo, initial_amount, horizon = build_suo(args)

    length = initial_amount + horizon

    testing_samples = []
    for x in range(args.testing_samples):
        path = suo.generate_random_traces([], length)[0]
        testing_samples.append(tuple(path))

    models_dict = {"IP": InvertedPendulum(), "MC": mc_model(horizon)}
    model = models_dict["MC"]
    model = mc_model(horizon)

    noisy_measurements = model.get_noisy_measurments(testing_samples, horizon)

    alarms = aggregted_alarms(testing_samples)
    target_risks = stats_true(horizon, initial_amount, testing_samples, suo)

    coarse = args.coarse
    mc = args.mc
    model_path = args.model_path
    stats_path = args.stats_path
    high_st = args.high_st

    conformal_risks = aggreagted_stats_conformal(
        args.high_st, noisy_measurements, coarse, mc, model_path, stats_path
    )


    (
        imc_risks_ref_splitting,
        imc_transition_counts_ref_splitting,
        imc_stopping_threshold_ref_splitting,
        imc_distances_ref_splitting,
    ) = aggregated_stats_imc(
        args.high_st,
        coarse,
        "refsplit",
        mc,
        stats_path,
        initial_amount,
        horizon,
        args,
        testing_samples,
    )

    print('imc_transition_counts_ref_splitting')
    print(imc_transition_counts_ref_splitting)


    imc_risks, imc_transition_counts, imc_stopping_threshold, imc_distances = (
        aggregated_stats_imc(
            args.high_st,
            coarse,
            "noref",
            mc,
            stats_path,
            initial_amount,
            horizon,
            args,
            testing_samples,
        )
    )

    print('imc_transition_counts')
    print(imc_transition_counts)

    (
        imc_risks_ref,
        imc_transition_counts_ref,
        imc_stopping_threshold_ref,
        imc_distances_ref,
    ) = aggregated_stats_imc(
        args.high_st,
        coarse,
        "ref",
        mc,
        stats_path,
        initial_amount,
        horizon,
        args,
        testing_samples,
    )

    print('imc_transition_counts_ref')
    print(imc_transition_counts_ref)

    fn_fp_comparison_model_based( 
        coarse,
        alarms,
        imc_risks,
        imc_risks_ref,
        target_risks,
        imc_risks_ref_splitting, 
        args.out)

    fn_fp_AUC_model_based(
        coarse,
        alarms,
        imc_risks,
        imc_risks_ref,
        target_risks,
        imc_risks_ref_splitting, 
        imc_transition_counts,
        imc_transition_counts_ref,
        imc_transition_counts_ref_splitting,
        args.out)

    regression_risks = aggregated_stats_regression(
        args.high_st,
        coarse,
        mc,
        model_path,
        stats_path,
        testing_samples,
        horizon,
        initial_amount,
    )
    
    

    test_data = {}
    test_data['model'] = args.mc
    test_data['coarse'] = args.coarse
    test_data['high_st'] = args.high_st
    test_data['stopping_threshold'] =  imc_stopping_threshold_ref
    test_data['horizon'] = horizon
    test_data['initial_amount'] = initial_amount
    test_data['testing_samples'] = testing_samples
    test_data['alarms'] = alarms 
    test_data['target_risks'] = target_risks
    test_data['imc_risks'] = imc_risks
    test_data['imc_risks_ref'] = imc_risks_ref
    test_data['imc_risks_ref_splitting'] = imc_risks_ref_splitting
    test_data['imc_distances'] = imc_distances
    test_data['imc_distances_ref'] = imc_distances_ref
    test_data['imc_distances_ref_splitting'] = imc_distances_ref_splitting
    test_data['regression_risks'] = regression_risks
    test_data['conformal_risks'] = conformal_risks
    test_data['imc_transition_counts'] = imc_transition_counts
    test_data['imc_transition_counts_ref'] = imc_transition_counts_ref
    test_data['imc_transition_counts_ref_splitting'] = imc_transition_counts_ref_splitting
   

    if args.coarse: 
        if args.high_st: 
            file_name = os.path.join(args.out, f"testdata_rq_3_{args.mc}_coarse_high_st.pkl")
        else: 
            file_name = os.path.join(args.out, f"testdata_rq_3_{args.mc}_coarse.pkl")
    else: 
        if args.high_st: 
            file_name = os.path.join(args.out, f"testdata_rq_3_{args.mc}_high_st.pkl")
        else:
            file_name = os.path.join(args.out, f"testdata_rq_3_{args.mc}.pkl")


    # Save dictionary
    with open(file_name, 'wb') as f:
        pickle.dump(test_data, f)

    fn_fp_comparison_model_based_vs_model_free(
        args.high_st,
        coarse,
        alarms,
        imc_risks_ref,
        imc_risks_ref_splitting,
        regression_risks,
        conformal_risks,
        target_risks,
        args.out) 

    if args.high_st == False:
        (
            target_auc,
            imc_results,
            imc_ref_results,
            imc_transition_counts_ref,
            imc_ref_splitting_results,
            imc_transition_counts_ref_splitting,
        ) = auc_graph_prep(
            imc_risks,
            target_risks,
            alarms,
            imc_risks_ref,
            imc_transition_counts_ref,
            imc_risks_ref_splitting,
            imc_transition_counts_ref_splitting,
        )
        plotting(
            coarse,
            target_auc,
            imc_results,
            imc_transition_counts,
            imc_ref_results,
            imc_transition_counts_ref,
            horizon,
            initial_amount,
            imc_ref_splitting_results,
            imc_transition_counts_ref_splitting,
            args.out,
        )


def build_learning_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Learning Parameters")

    group.add_argument(
        "--model_name",
        type=str,
        default="MC",
        help="Name of the conformal prediction model (first letters code).",
    )
    group.add_argument(
        "-s",
        "--testing_samples",
        type=int,
        default=500,
        help="Total number of samples used in learning",
    )
    group.add_argument(
        "--no-target",
        action="store_true",
        default=False,
        help="Do not use the target monitor",
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

    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to models",
    )

    parser.add_argument(
        "--stats-path",
        type=str,
        help="Path to stats",
    )

    parser.add_argument(
        "-c",
        "--coarse",
        action="store_true",
        default=False,
        help="If the leared model is coarse or not",
    )

    parser.add_argument(
        "-ht",
        "--high-st",
        action="store_true",
        default=False,
        help="If higher stopping threashold is used",
    )

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


# python -m premise.interval.rq_3 --mc evadeV-5-3 --stats-path /workspaces/premise/out/stats/2025-08-01_08-37-30 --model-path /workspaces/premise/out/models/2025-08-01_08-37-30