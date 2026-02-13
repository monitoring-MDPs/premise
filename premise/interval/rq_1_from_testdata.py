import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from tqdm import tqdm, trange
import matplotlib.ticker as ticker
import pickle
import numpy as np
#from premise.models import *
#from premise.interval.maximum_likelihood import *
from itertools import groupby
from operator import itemgetter
from matplotlib.ticker import LogLocator, FuncFormatter

from premise.interval.loading import build_suo_args_parser

#from premise.interval.conformence import test_monitor
#from premise.interval.interval import (
#    Trace,
#    create_monitor,
#)
#from premise.interval.maximum_likelihood import dict_to_pomdp, create_mle_monitor


def aggregted_alarms(testdata_rq_1):

    with open(testdata_rq_1, 'rb') as f:
        data = pickle.load(f)
    
    alarms = data['alarms']

    return alarms


def stats_true(testdata_rq_1):

    with open(testdata_rq_1, 'rb') as f:
        data = pickle.load(f)

    target_risks = data['target_risks']

    return target_risks


def aggregated_stats_imc(testdata_rq_1):

    with open(testdata_rq_1, 'rb') as f:
        data = pickle.load(f)

    imc_risks = data['imc_risks'] 
    imc_transition_counts = data['imc_transition_counts']
   
    return imc_risks, imc_transition_counts


def aggregated_stats_mc(testdata_rq_1):

    with open(testdata_rq_1, 'rb') as f:
        data = pickle.load(f)

    mc_risks = data['mc_risks'] 
    mc_transition_counts = data['mc_transition_counts']

    return mc_risks, mc_transition_counts



def fn_fp_comparison(coarse,
        alarms,
        imc_risks,
        mc_risks,
        target_risks,
        out_path, 
        high_st):

    
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
    target_risks = np.array(target_risks,  dtype=float)
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

        for key in mc_final_risks.keys(): 
                
                predictions_mc =  np.array(mc_final_risks[key], dtype=float) >= t

                fp_mc = np.sum((predictions_mc == 1) & (alarms == 0))
                fn_mc = np.sum((predictions_mc == 0) & (alarms == 1))
                    
                fnr_mc = fn_mc / actual_positives if actual_positives > 0 else 0.0
                fpr_mc = fp_mc / actual_negatives if actual_negatives > 0 else 0.0

                mc_fnr[key].append(fnr_mc)
                mc_fpr[key].append(fpr_mc)

    fig, ax = plt.subplots(figsize=(8, 5)) 

    auc_fnr_target = np.trapezoid(target_fnr, thresholds)
    auc_fpr_target = np.trapezoid(target_fpr, thresholds)


    ax.plot(thresholds, target_fnr, label=f'Target FNR: (AUC {auc_fnr_target:.3f})', color='black', linestyle= ':')
    ax.plot(thresholds, target_fpr, label=f'Target FPR: (AUC {auc_fpr_target:.3f})', color='black', linestyle= '--')
    
    imc_FNRs = []
    imc_FPRs = []

    imc_fnr_aucs = []
    imc_fpr_aucs = []


    for key in imc_final_risks.keys():
        imc_FNRs.append(imc_fnr[key]) 
        imc_FPRs.append(imc_fpr[key])
        imc_fnr_aucs.append(np.trapezoid(imc_fnr[key], thresholds))
        imc_fpr_aucs.append(np.trapezoid(imc_fpr[key], thresholds))

    imc_FNRs = np.array(imc_FNRs)    

    imc_fnr_mean = np.mean(imc_FNRs, axis=0)
    imc_fnr_std = np.std(imc_FNRs, axis=0)

    imc_fnr_aucs = np.array(imc_fnr_aucs)
    imc_fnr_auc = np.mean(imc_fnr_aucs, axis=0)

    imc_fpr_aucs = np.array(imc_fpr_aucs)
    imc_fpr_auc = np.mean(imc_fpr_aucs, axis=0)


    ax.plot(thresholds, imc_fnr_mean, label=f'iHMM FNR: (AUC {imc_fnr_auc:.3f})', color='red', linestyle= ':', marker ='P', markersize=4, markevery=18)

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

    ax.plot(thresholds, imc_fpr_mean, label=f'iHMM FPR: (AUC {imc_fpr_auc:.3f})', color='red', linestyle= '--', marker ='P', markersize=4, markevery=18)

    plt.fill_between(
    thresholds,
    imc_fpr_mean - imc_fpr_std,   
    imc_fpr_mean + imc_fpr_std,   
    color="red",
    alpha=0.2
    )


    mc_FNRs = []
    mc_FPRs = []

    mc_fnr_aucs = []
    mc_fpr_aucs = []


    for key in mc_final_risks.keys():
        mc_FNRs.append(mc_fnr[key]) 
        mc_FPRs.append(mc_fpr[key])
        mc_fnr_aucs.append(np.trapezoid(mc_fnr[key], thresholds))
        mc_fpr_aucs.append(np.trapezoid(mc_fpr[key], thresholds))

    mc_FNRs = np.array(mc_FNRs)    

    mc_fnr_mean = np.mean(mc_FNRs, axis=0)
    mc_fnr_std = np.std(mc_FNRs, axis=0)

    mc_fnr_aucs = np.array(mc_fnr_aucs)
    mc_fnr_auc = np.mean(mc_fnr_aucs, axis=0)

    mc_fpr_aucs = np.array(mc_fpr_aucs)
    mc_fpr_auc = np.mean(mc_fpr_aucs, axis=0)


    ax.plot(thresholds, mc_fnr_mean, label=f'HMM FNR: (AUC {mc_fnr_auc:.3f})', color='green', linestyle= ':', marker ='d', markersize=4, markevery=18)

    plt.fill_between(
    thresholds,
    mc_fnr_mean - mc_fnr_std,   
    mc_fnr_mean + mc_fnr_std,   
    color="green",
    alpha=0.2
    )

    mc_FPRs = np.array(mc_FPRs)

    mc_fpr_mean = np.mean(mc_FPRs, axis=0)
    mc_fpr_std = np.std(mc_FPRs, axis=0)

    ax.plot(thresholds, mc_fpr_mean, label=f'HMM FPR: (AUC {mc_fpr_auc:.3f})', color='green', linestyle= '--', marker ='d', markersize=4, markevery=18)

    plt.fill_between(
    thresholds,
    mc_fpr_mean - mc_fpr_std,   
    mc_fpr_mean + mc_fpr_std,   
    color="green",
    alpha=0.2
    )

    mc_fnr_mean = np.array(mc_fnr_mean)
    target_fnr = np.array(target_fnr)
    thresholds = np.array(thresholds)

    mask_2 = target_fnr < mc_fnr_mean

    count = 0 

    indices_2 = np.where(mask_2)[0]
    for k, g in groupby(enumerate(indices_2), lambda i: i[0]-i[1]):
        count +=1 
        group = list(map(itemgetter(1), g))
        start = thresholds[group[0]]
        end = thresholds[group[-1]]
        if count < 1:
            ax.axvspan(start, end, color='peachpuff', alpha=0.5,  label='HMM mean FNR > Target FNR')
        else:
            ax.axvspan(start, end, color='peachpuff', alpha=0.5) 

    ax.set_xlabel('Threshold', fontsize=18)
    ax.set_ylabel('Rate', fontsize=18)
    ax.legend(fontsize=18)
    ax.grid(True)
    plt.tight_layout()


    if coarse:
        if high_st:
            fig.savefig(
                    f"{out_path}/rq_1_{args.mc}_high_st_coarse_FN_FP_multi.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
        else: 
            fig.savefig(
                    f"{out_path}/rq_1_{args.mc}_coarse_FN_FP_multi.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
    else:
        if high_st: 
            fig.savefig(
                    f"{out_path}/rq_1_{args.mc}_high_st_FN_FP_multi.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
        else: 
            print(f"{out_path}/rq_1_{args.mc}_FN_FP_multi.pdf")
            fig.savefig(
                    f"{out_path}/rq_1_{args.mc}_FN_FP_multi.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )

    plt.show()





def distance_graph(
    coarse,
    target_risks,
    testing_samples,
    imc_risks,
    imc_transition_counts,
    imc_risks_ref,
    imc_transition_counts_ref,
    imc_risks_refsplit,
    imc_transition_counts_refsplit,
    out_path,
    high_st
):

    avg = 0 
    for x in imc_transition_counts.keys(): 
        print(x, imc_transition_counts[x][-1])
        avg += imc_transition_counts[x][-1]

    print(np.mean(avg))


    avg = 0 
    for x in imc_transition_counts_ref.keys(): 
        print(x, imc_transition_counts_ref[x][-1])
        avg += imc_transition_counts_ref[x][-1]

    print(np.mean(avg))

    avg = 0 
    for x in imc_transition_counts_refsplit.keys(): 
        print(x, imc_transition_counts_refsplit[x][-1])
        avg += imc_transition_counts_refsplit[x][-1]
    
    print(np.mean(avg))

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
        distance_stats[key] = total_distance / testing_samples

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
        ref_distance_stats[key] = total_distance / testing_samples

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
        refsplit_distance_stats[key] = total_distance / testing_samples

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

    log = True
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
        label=f"No refinement: (Final distance: {np.mean(final_distances):.3f})",
        linewidth=3,
        linestyle="--",
        marker ='P', markersize=10, markevery=18
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

    
    for x in range(0,10):
            if len(ref_transitions_data[x]) > 20: 
                row = ref_transitions_data[x]
                ref_transitions_data[x] = [row[0]] + row[4:-1:5] + [row[-1]]

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
        label=f"Refinement: (Final distance: {np.mean(ref_final_distances):.3f})",
        linewidth=3,
        linestyle="--",
        marker ='o', markersize=10, markevery=18
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

     
    for x in range(0,10):
        if len(refsplit_transitions_data[x]) > 20:
            row = refsplit_transitions_data[x]
            refsplit_transitions_data[x] = [row[0]] + row[4:-1:5] + [row[-1]]

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
    """ ax.plot(
        refsplit_x_values,
        refsplit_mean_distance,
        color="aqua",
        label=f"Refinement with splitting, (Average final distance: {np.mean(refsplit_final_distances):.3f})",
        linewidth=3,
        linestyle="--",
        marker ='s', markersize=10, markevery=18
    )

    # Add shaded area for spread
    ax.fill_between(
        refsplit_x_values,
        refsplit_mean_distance - refsplit_std_distance,
        refsplit_mean_distance + refsplit_std_distance,
        alpha=0.2,
        color="aqua",
    ) """

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((4, 4))  # Force 10^4 scale
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="both", labelsize=30)
    ax.xaxis.get_offset_text().set_size(30)

    ax.set_xlabel("Explored states", fontsize=40)
    ax.set_ylabel("Distance to Target", fontsize=40)
    ax.legend(loc="upper right", fontsize=36)
    if log:
        plt.yscale("log")
        
    else:
        plt.yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{y:g}")
        )
        ax.set_ylim(1e-3, 1) 
    ax.grid(True)
    plt.subplots_adjust(bottom=0.25)

    plt.tight_layout()
    if coarse:
        if high_st:
            plt.savefig(
                f"{out_path}/rq_1_{args.mc}_high_st_coarse_distance_to_RRF.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                    f"{out_path}/rq_1_{args.mc}_coarse_distance_to_RRF.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
    else:
        if high_st:
            plt.savefig(
                f"{out_path}/rq_1_{args.mc}_high_st_distance_to_RRF.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        else: 
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
    mc_risks,
    out_path,
    high_st,
    mc_model
):

    plt.figure()
    plt.plot([0, 1], [0, 1], "--", color="black")

    mc_ys = []

    for key in mc_risks.keys():
        mc_ys.append(int(key.split("-")[1]))


    hmm_under = 0 
    hmm_over = 0 
    hmm_equal = 0 

    for key in mc_risks.keys():
        if key.split("-")[1] == str(max(mc_ys)):
            for x in range(len(target_risks)): 
                if mc_risks[key][x] < target_risks[x]:
                    hmm_under += 1
                if mc_risks[key][x] > target_risks[x]:
                    hmm_over +=1 
                if mc_risks[key][x] == target_risks[x]:
                    hmm_equal +=1

                       

    count_u = 0 
    count_o = 0 

    
    for key in mc_risks.keys():
        if key.split("-")[1] == str(max(mc_ys)):
            for x in range(len(target_risks)): 
                if target_risks[x] != 0.0 and mc_risks[key][x] != 0.0: 
                    if mc_risks[key][x] < target_risks[x]:
                            if count_u < 1:
                                plt.scatter(
                                        mc_risks[key][x],
                                        target_risks[x],
                                        color="greenyellow",
                                        marker="s",
                                        s=8,
                                        label=f"HMM under ({((hmm_under/(hmm_under + hmm_over+ hmm_equal))*100):.2f}%)",       
                                )
                                count_u += 1 
                            else: 
                                plt.scatter(
                                        mc_risks[key][x],
                                        target_risks[x],
                                        color="greenyellow",
                                        marker="s",
                                        s=8,
                                )
                    if mc_risks[key][x] > target_risks[x]:
                        if count_o < 1:
                            plt.scatter(
                                    mc_risks[key][x],
                                    target_risks[x],
                                    color="darkgreen",
                                    marker="s",
                                    s=8,
                                    label=f"HMM over ({((hmm_over/(hmm_under + hmm_over + hmm_equal))*100):.2f}%)",
                            )
                            count_o += 1 
                        else: 
                            plt.scatter(
                                    mc_risks[key][x],
                                    target_risks[x],
                                    color="darkgreen",
                                    marker="s",
                                    s=8,
                            )


    imc_ys = []

    for key in imc_risks.keys():
        imc_ys.append(int(key.split("-")[1]))

    ihmm_under = 0 
    ihmm_over = 0 
    ihmm_even = 0 

    

    for key in imc_risks.keys():
        if key.split("-")[1] == str(max(imc_ys)):
            for x in range(len(target_risks)): 
                if imc_risks[key][x] < target_risks[x]:
                        ihmm_under += 1
                if imc_risks[key][x] > target_risks[x]:
                        ihmm_over += 1
                if imc_risks[key][x] == target_risks[x]:
                        ihmm_even += 1
    

    count_u = 0 
    count_o = 0 

    for key in imc_risks.keys():
        if key.split("-")[1] == str(max(imc_ys)):
            for x in range(len(target_risks)):
                if  target_risks[x] != 0.0 and imc_risks[key][x] != 0.0: 
                    if imc_risks[key][x] < target_risks[x]:
                            if count_u < 1:
                                plt.scatter(
                                        imc_risks[key][x],
                                        target_risks[x],
                                        color="orange",
                                        marker="o",
                                        s=8,
                                        label=f"iHMM under ({((ihmm_under/(ihmm_under + ihmm_over+ ihmm_even))* 100):.2f} %)",
                                )
                                count_u += 1 
                            else: 
                                plt.scatter(
                                        imc_risks[key][x],
                                        target_risks[x],
                                        color="orange",
                                        marker="o",
                                        s=8
                                )
                    if imc_risks[key][x] > target_risks[x]:
                            if count_o < 1:
                                plt.scatter(
                                        imc_risks[key][x],
                                        target_risks[x],
                                        color="red",
                                        marker="o",
                                        s=8,
                                        label=f"iHMM over ({((ihmm_over / (ihmm_under + ihmm_over + ihmm_even))*100):.2f}%)",
                                )
                                count_o += 1 
                            else: 
                                plt.scatter(
                                        imc_risks[key][x],
                                        target_risks[x],
                                        color="red",
                                        marker="o", 
                                        s=8
                                )

    
    plt.legend(fontsize=18)

    plt.xlabel("iHMM and HMM risks", fontsize=18)
    plt.ylabel("Target risks", fontsize=18)
    plt.tick_params(axis="both", labelsize=12)

    plt.tight_layout()


    import os
    os.makedirs(out_path, exist_ok=True)

    if coarse:
        if high_st:
            plt.savefig(
                f"{out_path}/rq_1_{mc_model}_high_st_coarse_overestimation.pdf",
                dpi=300,
            )
        else:
            plt.savefig(
                    f"{out_path}/rq_1_{mc_model}_coarse_overestimation.pdf",
                    dpi=300,
                )
    else:
        if high_st:
            plt.savefig(
                f"{out_path}/rq_1_{mc_model}_high_st_overestimation.pdf",
                dpi=300,
            )
        else: 
            plt.savefig(
                    f"{out_path}/rq_1_{mc_model}_overestimation.pdf",
                    dpi=300,
                )
    plt.show()


def roc_curve_imc_mc(
    coarse,
    alarms,
    imc_risks,
    mc_risks,
    target_risks,
    out_path,
    high_st
):

    # Extract unique experiment numbers from iHMM data
    imc_experiment_numbers = set()
    for key in imc_risks.keys():
        exp_num = int(key.split("-")[0])
        imc_experiment_numbers.add(exp_num)

    # Extract unique experiment numbers from HMM data
    mc_experiment_numbers = set()
    for key in mc_risks.keys():
        exp_num = int(key.split("-")[0])
        mc_experiment_numbers.add(exp_num)

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

    plt.figure()
    fig, ax = plt.subplots(figsize=(16, 12))


    fpr, tpr, thresholds = metrics.roc_curve(alarms, target_risks)
    target_auc = metrics.auc(fpr, tpr)

    ax.plot(
        fpr,
        tpr,
        color="black",
        label=f"Target Monitor, AUC = {target_auc:.3f}",
        linewidth=5,
        linestyle="-",
        marker="D",
    )

    # iHMM MEAN PERFORMANCE
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
        label=f"iHMM, (Mean AUC = {mean_auc:.3f})",
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

    
    # HMM MEAN PERFORMANCE
    mc_roc_data_mean = {}

    for key in mc_final_risks.keys():
        fpr, tpr, thresholds = metrics.roc_curve(alarms, mc_final_risks[key])
        roc_auc = metrics.auc(fpr, tpr)
        mc_roc_data_mean[key] = [fpr, tpr, roc_auc]

    mean_fpr = np.linspace(0, 1, 100)

    mc_tprs = []
    mc_aucs = []

    for key in mc_roc_data_mean.keys():
        interp_tpr = np.interp(
            mean_fpr, mc_roc_data_mean[key][0], mc_roc_data_mean[key][1]
        )
        mc_aucs.append(mc_roc_data_mean[key][2])
        interp_tpr[0] = 0.0
        mc_tprs.append(interp_tpr)

    mean_tpr = np.mean(mc_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(mc_aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="chartreuse",
        label=f"HMM, (Mean AUC = {mean_auc:.3f})",
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
        color="chartreuse",
        alpha=0.2,
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
                f"{out_path}/rq1_{args.mc}_high_st_coarse_ROC_HMM_iHMM_comparison.pdf",
                dpi=300,
                bbox_inches="tight",
            ) 
        else:
            plt.savefig(
                f"{out_path}/rq1_{args.mc}_coarse_ROC_HMM_iHMM_comparison.pdf",
                dpi=300,
                bbox_inches="tight",
            )
    else:
        if high_st:
            plt.savefig(
                f"{out_path}/rq1_{args.mc}_high_st_ROC_HMM_iHMM_comparison.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                f"{out_path}/rq1_{args.mc}_ROC_HMM_iHMM_comparison.pdf",
                dpi=300,
                bbox_inches="tight",
            )
    plt.show()


def main_imc(args: argparse.Namespace):
    #setup_logging("rq1_from_testdata:" + args.mc)


    with open(args.testdata_rq_1, 'rb') as f:
        data = pickle.load(f)

    if 'high_st' in data.keys(): 
        high_st = data['high_st']
    else:   
        high_st = args.high_st  
    
    coarse = data['coarse']
    model = data['model']
    testing_samples = data['testing_samples']
    testing_samples_amount = data['testing_samples_amount']
    alarms = data['alarms']
    target_risks = data['target_risks']
    imc_risks = data['imc_risks']

    mc_transition_counts = data['mc_transition_counts']
    imc_transition_counts = data['imc_transition_counts']
    mc_risks = data['mc_risks']
    imc_risks_ref = data['imc_risks_ref']
    imc_transition_counts_ref = data['imc_transition_counts_ref']
    imc_risks_refsplit = data['imc_risks_refsplit']
    imc_transition_counts_refsplit = data['imc_transition_counts_refsplit']




    if len(imc_risks) == 0 or len(mc_risks) == 0:
        print("No data for IMC or MC risks, skipping graph generation.")
        return

    
    overestimation_graph(
        coarse,
        target_risks,
        imc_risks,
        mc_risks,
        args.out,
        high_st, 
        args.mc
    )

    roc_curve_imc_mc(
        coarse,
        alarms,
        imc_risks,
        mc_risks,
        target_risks,
        args.out,
        high_st
    ) 
    
    fn_fp_comparison(
        coarse,
        alarms,
        imc_risks,
        mc_risks,
        target_risks,
        args.out,
        high_st
    )  
    

    distance_graph(
        coarse,
        target_risks,
        testing_samples_amount,
        imc_risks,
        imc_transition_counts,
        imc_risks_ref,
        imc_transition_counts_ref,
        imc_risks_refsplit,
        imc_transition_counts_refsplit,
        args.out,
        high_st
    )   


def testing_argsparser():
    parser = argparse.ArgumentParser(description="Learn an IMC")
    build_suo_args_parser(parser)

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (can be used multiple times)",
    )

    parser.add_argument(
        "--testdata_rq_1", type=str, help="Path to test data")

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
        default="/workspaces/premise/premise/results",
        help="Output path for the results",
    )

    return parser


if __name__ == "__main__":
    parser = testing_argsparser()
    args = parser.parse_args()
    main_imc(args)



# python -m premise.interval.rq_1_from_testdata --testdata_rq_1 /workspaces/premise/premise/results/testdata_rq_1_evadeV-6-3-coarse_coarse.pkl --mc evadeV-6-3-coarse  --out /workspaces/premise/premise/results

## python -m premise.interval.rq_1_from_testdata --testdata_rq_1 /workspaces/premise/premise/results/testdata_rq_1_airportA-7-10-10.pkl --mc airportA-7-10-10 --out /workspaces/premise/premise/results


#python -m premise.interval.rq_1_from_testdata --testdata_rq_1 /workspaces/premise/premise/results/testdata_rq_1_evadeV-5-3.pkl --mc evadeV-5-3 --out workspaces/premise/premise/results


#python -m premise.interval.rq_1_from_testdata --testdata_rq_1 /workspaces/premise/premise/analysis/testdata_rq_1_unlikely-15.pkl --mc unlikely-15 --out /workspaces/premise/premise/results