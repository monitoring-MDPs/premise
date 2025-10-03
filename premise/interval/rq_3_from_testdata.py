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
import numpy as np
import argparse
from matplotlib import pyplot as plt
import math

from premise.interval.loading import (
    build_suo_args_parser

)

def fn_fp_comparison_model_based(
        coarse,
        alarms,
        imc_risks,
        imc_risks_ref,
        target_risks,
        imc_risks_ref_splitting, 
        out_path,
        high_st):

    
    # iHMM
    imc_final_risks = {}

    ys = []
    for key in imc_risks.keys():
        print(key)
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



    for x in range(len(target_fnr)):
        print(f'FOR : {x}')
        print(target_fnr[x])
        print(imc_fnr_mean[x])



    ax.plot(thresholds, imc_fnr_mean, label=f'No refinement mean FNR: (AUC {imc_fnr_auc:.3f})', color='red', linestyle= ':', marker ='P', markersize=4, markevery=32)

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

    ax.plot(thresholds, imc_fpr_mean, label=f'No refinement mean FPR: (AUC {imc_fpr_auc:.3f})', color='red', linestyle= '--', marker ='P', markersize=4, markevery=32)

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


    ax.plot(thresholds, imc_ref_fnr_mean, label=f'Refinement mean FNR: (AUC {imc_ref_fnr_auc:.3f})', color='blue', linestyle= ':', marker ='o', markersize=4, markevery=32)

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

    ax.plot(thresholds, imc_ref_fpr_mean, label=f'Refinement mean FPR: (AUC {imc_ref_fpr_auc:.3f})', color='blue', linestyle= '--', marker ='o', markersize=4, markevery=32)

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


    ax.plot(thresholds, imc_ref_splitting_fnr_mean, label=f'Refinement with Splitting mean FNR: (AUC {imc_ref_splitting_fnr_auc:.3f})', color='aqua', linestyle= ':', marker ='s', markersize=4, markevery=32)

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

    ax.plot(thresholds, imc_ref_splitting_fpr_mean, label=f'Refinement with Splitting mean FPR: (AUC {imc_ref_splitting_fpr_auc:.3f})', color='aqua', linestyle= '--', marker ='s', markersize=4, markevery=32)

    plt.fill_between(
    thresholds,
    imc_ref_splitting_fpr_mean - imc_ref_splitting_fpr_std,   
    imc_ref_splitting_fpr_mean + imc_ref_splitting_fpr_std,   
    color="aqua",
    alpha=0.2
    )


    ax.set_xlabel('Threshold')
    ax.set_ylabel('Rate')
    ax.legend(loc="right", fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


    if coarse:
        if high_st:
            fig.savefig(
                    f"{out_path}/rq_2_{args.mc}_high-st_coarse_FN_FP_model_based.pdf",
                    dpi=300,
                    bbox_inches="tight",
            )
        else:
            fig.savefig(
                    f"{out_path}/rq_2_{args.mc}_coarse_FN_FP_model_based.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
    else:
        if high_st:
            fig.savefig(
                    f"{out_path}/rq_2_{args.mc}_high-st_FN_FP_model_based.pdf",
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
        out_path, 
        high_st):

    
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

    ax.plot([0, int(y_range)], [auc_fnr_target, auc_fnr_target], label = f'Target FNR AUC (Final AUC: {float(auc_fnr_target):.3f})', color = 'black', linestyle= '--')
    ax.plot([0, int(y_range)], [auc_fpr_target, auc_fpr_target], label = f'Target FPR AUC (Final AUC: {float(auc_fpr_target):.3f})', color = 'black', linestyle= ':') 

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
    mean_no_ref_x = np.nanmean(no_ref_x, axis=0)

    fnr_no_rf_y = np.array(fnr_no_rf_y)
    mean_fnr_no_rf_y = np.nanmean(fnr_no_rf_y, axis=0)
    std_fnr_no_rf_y = np.nanstd(fnr_no_rf_y, axis=0)

    fpr_no_rf_y = np.array(fpr_no_rf_y)
    mean_fpr_no_rf_y = np.nanmean(fpr_no_rf_y, axis=0)
    std_fpr_no_rf_y = np.nanstd(fpr_no_rf_y, axis=0)

    print('NO REF MEAN X AXIS')
    for x in no_ref_x: 
        if not math.isnan(x[-1]):
            print(x[-1])
    print(mean_no_ref_x[-1])

    ax.plot(mean_no_ref_x, mean_fnr_no_rf_y, label = f'No refinement FNR AUC (Final AUC: {float(mean_fnr_no_rf_y[-1]):.3f})', color='red', linestyle= '--' , marker ='P', markersize=4, markevery=1)
    ax.plot(mean_no_ref_x, mean_fpr_no_rf_y, label = f'No refinement FPR AUC (Final AUC: {float(mean_fpr_no_rf_y[-1]):.3f})', color='red', linestyle= ':' , marker ='P', markersize=4, markevery=1)

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

    max_len = 0 

    for x in ref_x:
         if len(x) > max_len: 
              max_len = len(x)
              
    ref_x_padded = [r + [np.nan]*(max_len - len(r)) for r in ref_x]
    arr = np.array(ref_x_padded, dtype=float)
    mean_ref_x = np.nanmean(arr, axis=0)

    fnr_rf_y_flat = [[float(v[0]) for v in row] for row in fnr_rf_y]
    fnr_rf_y_padded = [r + [np.nan]*(max_len - len(r)) for r in fnr_rf_y_flat]
    fnr_rf_y_padded =  np.array(fnr_rf_y_padded)

    mean_fnr_rf_y = np.nanmean(fnr_rf_y_padded, axis=0)
    std_fnr_rf_y = np.nanstd(fnr_rf_y_padded, axis=0)

    fpr_rf_y_flat = [[float(v[0]) for v in row] for row in fpr_rf_y]
    fpr_rf_y_padded = [r + [np.nan]*(max_len - len(r)) for r in fpr_rf_y_flat]
    fpr_rf_y_padded =  np.array(fpr_rf_y_padded)

    mean_fpr_rf_y = np.nanmean(fpr_rf_y_padded, axis=0)
    std_fpr_rf_y = np.nanstd(fpr_rf_y_padded, axis=0)


    print('REF MEAN X AXIS')
    for x in ref_x_padded: 
        if not math.isnan(x[-1]):
            print(x[-1])
    print(mean_ref_x[-1])

    ax.plot(mean_ref_x, mean_fnr_rf_y, label = f'Refinement FNR AUC  (Final AUC: {float(mean_fnr_rf_y[-1]):.3f})', color='blue', linestyle= '--' , marker ='o', markersize=4, markevery=1)
    ax.plot(mean_ref_x, mean_fpr_rf_y, label = f'Refinement FPR AUC  (Final AUC: {float(mean_fpr_rf_y[-1]):.3f})', color='blue', linestyle= ':', marker ='o', markersize=4, markevery=1)

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

    max_len = 0 

    for x in ref_split_x:
         if len(x) > max_len: 
              max_len = len(x)
    
    ref_split_x_padded = [r + [np.nan]*(max_len - len(r)) for r in ref_split_x]

    arr = np.array(ref_split_x_padded, dtype=float)
    mean_ref_split_x = np.nanmean(arr, axis=0)

    fnr_rf_split_y_flat = [[float(v[0]) for v in row] for row in fnr_rf_split_y]
    fnr_rf_split_y_padded = [r + [np.nan]*(max_len - len(r)) for r in fnr_rf_split_y_flat]
    fnr_rf_split_y_padded =  np.array(fnr_rf_split_y_padded)

    mean_fnr_rf_split_y = np.nanmean(fnr_rf_split_y_padded, axis=0)
    std_fnr_rf_split_y = np.nanstd(fnr_rf_split_y_padded, axis=0)

    fpr_rf_split_y_flat = [[float(v[0]) for v in row] for row in fpr_rf_split_y]
    fpr_rf_split_y_padded = [r + [np.nan]*(max_len - len(r)) for r in fpr_rf_split_y_flat]
    fpr_rf_split_y_padded =  np.array(fpr_rf_split_y_padded)

    mean_fpr_rf_split_y = np.nanmean(fpr_rf_split_y_padded, axis=0)
    std_fpr_rf_split_y = np.nanstd(fpr_rf_split_y_padded, axis=0)

    print('REF SPLIT X AXIS')
    for x in ref_split_x_padded: 
        if not math.isnan(x[-1]):
            print(x[-1])
    print(mean_ref_split_x[-1])

    ax.plot(mean_ref_split_x, mean_fnr_rf_split_y, label = f'Refinement with Splitting FNR AUC  (Final AUC: {float(mean_fnr_rf_split_y[-1]):.3f})', color='aqua', linestyle= '--', marker ='s', markersize=4, markevery=1)
    ax.plot(mean_ref_split_x, mean_fpr_rf_split_y, label = f'Refinement with Splitting FPR AUC   (Final AUC: {float(mean_fpr_rf_split_y[-1]):.3f})', color='aqua', linestyle= ':', marker ='s', markersize=4, markevery=1)

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

    ax.set_ylabel('AUC' , fontsize=13)
    ax.set_xlabel('Explored States' , fontsize=13)
    ax.legend(loc="right", fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


    if coarse:
        if high_st: 
            fig.savefig(
                    f"{out_path}/rq_2_{args.mc}_high-st_coarse_FN_FP_AUC_model_based.pdf",
                    dpi=300,
                    bbox_inches="tight",
            )
             
        else: 
            fig.savefig(
                    f"{out_path}/rq_2_{args.mc}_coarse_FN_FP_AUC_model_based.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
    else:
        if high_st:
            fig.savefig(
                    f"{out_path}/rq_2_{args.mc}_high-st_FN_FP_AUC_model_based.pdf",
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


    ax.plot(thresholds, target_fnr, label=f'Target FNR: (AUC {float(auc_fnr_target):.3f})', color='black', linestyle= ':')
    ax.plot(thresholds, target_fpr, label=f'Target FPR: (AUC {float(auc_fpr_target):.3f})', color='black', linestyle= '--')

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


    ax.plot(thresholds, imc_ref_fnr_mean, label=f'Refinement mean FNR: (AUC {imc_ref_fnr_auc:.3f})', color='blue', linestyle= ':', marker ='o', markersize=4, markevery=25)

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

    ax.plot(thresholds, imc_ref_fpr_mean, label=f'Refinement mean FPR: (AUC {imc_ref_fpr_auc:.3f})', color='blue', linestyle= '--', marker ='o', markersize=4, markevery=25)

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


    ax.plot(thresholds, imc_ref_splitting_fnr_mean, label=f'Refinement with Splitting mean FNR: (AUC {imc_ref_splitting_fnr_auc:.3f})', color='aqua', linestyle= ':', marker ='s', markersize=4, markevery=25)

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

    ax.plot(thresholds, imc_ref_splitting_fpr_mean, label=f'Refinement with Splitting mean FPR: (AUC {imc_ref_splitting_fpr_auc:.3f})', color='aqua', linestyle= '--', marker ='s', markersize=4, markevery=25)

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


    ax.plot(thresholds, reg_fnr_mean, label=f'Regression mean FNR: (AUC {reg_fnr_auc:.3f})', color='pink', linestyle= ':',  marker ='D', markersize=4, markevery=25)

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

    ax.plot(thresholds, reg_fpr_mean, label=f'Regression mean FPR: (AUC {reg_fpr_auc:.3f})', color='pink', linestyle= '--',  marker ='D', markersize=4, markevery=25)

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


    ax.plot(thresholds, conformal_fnr_mean, label=f'Conformal prediction mean FNR: (AUC {conformal_fnr_auc:.3f})', color='orange', linestyle= ':',  marker ='x', markersize=4, markevery=25)

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

    ax.plot(thresholds, conformal_fpr_mean, label=f'Conformal prediction mean FPR: (AUC {conformal_fpr_auc:.3f})', color='orange', linestyle= '--',  marker ='x', markersize=4, markevery=25)

    plt.fill_between(
    thresholds,
    conformal_fpr_mean - conformal_fpr_std,   
    conformal_fpr_mean + conformal_fpr_std,   
    color="orange",
    alpha=0.2
    )

    ax.set_xlabel('Threshold', fontsize=14)
    ax.set_ylabel('Rate', fontsize=14)
    ax.legend(loc="right", fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


    if coarse:
        if high_st:
             fig.savefig(
                    f"{out_path}/rq_3_{args.mc}_high-st_coarse_FN_FP_model_based_vs_model_free.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
        else:
            fig.savefig(
                    f"{out_path}/rq_3_{args.mc}_coarse_FN_FP_model_based_vs_model_free.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
    else:
        if high_st:
             fig.savefig(
                    f"{out_path}/rq_3_{args.mc}_high-st_FN_FP_model_based_vs_model_free.pdf",
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

    with open(args.testdata_rq_3, 'rb') as f:
        data = pickle.load(f)

    mc = data['model']
    initial_amount = data['initial_amount']
    horizon = data['horizon']
    alarms = data['alarms']
    target_risks = data['target_risks']
    coarse = data['coarse']
    high_st = data['high_st']

    imc_risks = data['imc_risks'] 
    imc_risks_ref = data['imc_risks_ref']
    imc_risks_ref_splitting = data['imc_risks_ref_splitting']
    regression_risks = data['regression_risks'] 
    conformal_risks = data['conformal_risks']
    imc_transition_counts = data['imc_transition_counts'] 
    imc_transition_counts_ref = data['imc_transition_counts_ref']
    imc_transition_counts_ref_splitting = data['imc_transition_counts_ref_splitting']

    fn_fp_comparison_model_based( 
        coarse,
        alarms,
        imc_risks,
        imc_risks_ref,
        target_risks,
        imc_risks_ref_splitting, 
        args.out, 
        high_st)
    
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
        args.out, 
        high_st) 
    
    fn_fp_comparison_model_based_vs_model_free(
        high_st,
        coarse,
        alarms,
        imc_risks_ref,
        imc_risks_ref_splitting,
        regression_risks,
        conformal_risks,
        target_risks,
        args.out)  

   

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
        "--testdata_rq_3", type=str, help="Path to test data")

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

#python -m premise.interval.rq_3_from_testdata --testdata_rq_3 /workspaces/premise/premise/results/testdata_rq_3_airportA-7-10-10.pkl --mc airportA-7-10-10
#python -m premise.interval.rq_3_from_testdata --testdata_rq_3 /workspaces/premise/premise/results/testdata_rq_3_airportA-7-10-10_high_st.pkl --mc airportA-7-10-10

#python -m premise.interval.rq_3_from_testdata  --testdata_rq_3  /workspaces/premise/premise/results/testdata_rq_3_evadeV-6-3_high_st.pkl --mc evadeV-6-3
#python -m premise.interval.rq_3_from_testdata  --testdata_rq_3  /workspaces/premise/premise/results/testdata_rq_3_evadeV-6-3.pkl --mc evadeV-6-3

