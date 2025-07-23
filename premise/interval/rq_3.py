import argparse
import logging
from pathlib import Path
import glob 
import re
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle
from stormpy import AddUncertaintyExact, Rational
from premise.interval.utils import logger
from sklearn import metrics
import matplotlib.ticker as ticker


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

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle
from stormpy import AddUncertaintyExact, Rational
from premise.interval.utils import logger

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


from premise.interval.interval import (
    stormpy_imdp_to_ipomdp,
    stormpy_exact_pomdp_to_mdp,
)
from premise.interval.model_free.regression_model import prep_trace_for_regression
from premise.interval.loading import (
    build_imc_loading_args_parser,
    build_suo,
    build_suo_args_parser,
    load_imc,
)
from premise.interval.conformence import random_sample_monitor_test, test_monitor
from premise.interval.interval import (
    Samples,
    Trace,
    create_monitor,
    build_monitor_from_model,
)

def stats_true(horizon, initial_amount, testing_samples, suo): 


    target_risks = []

    for trace in tqdm(testing_samples):
        sub_trace: Trace = trace[:initial_amount]
      
        target_risk = test_monitor(
                    suo.create_target_monitor(),
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


def aggregated_stats_imc(high_st, coarse, method, mc, model_path, stats_path, initial_amount, horizon, args, testing_samples): 

    imc_risks = {}

    imc_transition_counts = {}
    imc_distances = {}

    for x in range(1,11):
    #for x in range(5,7):
        print(f'Experiment number {x}')
        if coarse: 
            if high_st:
                if method == 'noref':
                    statistics = np.load(f'{stats_path}/high-st-{mc}-coarse_norefinement-stats-{x}.npy', allow_pickle=True)
                elif method == 'ref':
                    statistics = np.load(f'{stats_path}/high-st-{mc}-coarse_refinement-stats-{x}.npy', allow_pickle=True)
                elif method == 'refsplit':
                    statistics = np.load(f'{stats_path}/high-st-{mc}-coarse_refsplitinement-stats-{x}.npy', allow_pickle=True)
            else:
                if method == 'noref':
                    statistics = np.load(f'{stats_path}/{mc}-coarse_norefinement-stats-{x}.npy', allow_pickle=True)
                elif method == 'ref':
                    statistics = np.load(f'{stats_path}/{mc}-coarse_refinement-stats-{x}.npy', allow_pickle=True)
                elif method == 'refsplit':
                    statistics = np.load(f'{stats_path}/{mc}-coarse_refsplitinement-stats-{x}.npy', allow_pickle=True)
        else: 
            if high_st:
                if method == 'noref':
                    statistics = np.load(f'{stats_path}/high-st-{mc}-comp-noref-stats-{x}.npy', allow_pickle=True)
                elif method == 'ref':
                    statistics = np.load(f'{stats_path}/high-st-{mc}-comp-ref-stats-{x}.npy', allow_pickle=True)
                elif method == 'refsplit':
                    statistics = np.load(f'{stats_path}/high-st-{mc}-comp-refsplit-stats-{x}.npy', allow_pickle=True)
            else:
                if method == 'noref':
                    statistics = np.load(f'{stats_path}/{mc}-comp-noref-stats-{x}.npy', allow_pickle=True)
                elif method == 'ref':
                    statistics = np.load(f'{stats_path}/{mc}-comp-ref-stats-{x}.npy', allow_pickle=True)
                elif method == 'refsplit':
                    statistics = np.load(f'{stats_path}/{mc}-comp-refsplit-stats-{x}.npy', allow_pickle=True)


        obj = statistics.item()     
        imc_transition_count = obj['transitions_learned']
        stopping_threashold = obj['args']['stopping_threshold']
        distances = obj['distances']
        imc_distances[str(x)] = distances
        imc_transition_counts[str(x)]= imc_transition_count


        for y in range(1, len(imc_transition_count) +1): 
            if coarse:
                if high_st:
                    if method == 'noref':
                        initial_distribution = f'{model_path}/high-st-{mc}-coarse-comp-noref-{x}-{y}-initial_interval.npy'
                        transition_intervals = f'{model_path}/high-st-{mc}-coarse-comp-noref-{x}-{y}-interval.npy'
                    elif method == 'ref':
                        initial_distribution = f'{model_path}/high-st-{mc}-coarse-comp-ref-{x}-{y}-initial_interval.npy'
                        transition_intervals = f'{model_path}/high-st-{mc}-coarse-comp-ref-{x}-{y}-interval.npy'
                    elif method == 'refsplit':
                        initial_distribution = f'{model_path}/high-st-{mc}-coarse-comp-refsplit-{x}-{y}-initial_interval.npy'
                        transition_intervals = f'{model_path}/high-st-{mc}-coarse-comp-refsplit-{x}-{y}-interval.npy'
                else:
                    if method == 'noref':
                        initial_distribution = f'{model_path}/{mc}-coarse-comp-noref-{x}-{y}-initial_interval.npy'
                        transition_intervals = f'{model_path}/{mc}-coarse-comp-noref-{x}-{y}-interval.npy'
                    elif method == 'ref':
                        initial_distribution = f'{model_path}/{mc}-coarse-comp-ref-{x}-{y}-initial_interval.npy'
                        transition_intervals = f'{model_path}/{mc}-coarse-comp-ref-{x}-{y}-interval.npy'
                    elif method == 'refsplit':
                        initial_distribution = f'{model_path}/{mc}-coarse-comp-refsplit-{x}-{y}-initial_interval.npy'
                        transition_intervals = f'{model_path}/{mc}-coarse-comp-refsplit-{x}-{y}-interval.npy'
            else: 
                if high_st: 
                    if method == 'noref':
                        initial_distribution = f'{model_path}/high-st-{mc}-comp-noref-{x}-{y}-initial_interval.npy'
                        transition_intervals = f'{model_path}/high-st-{mc}-comp-noref-{x}-{y}-interval.npy'
                    elif method == 'ref':
                        initial_distribution = f'{model_path}/high-st-{mc}-comp-ref-{x}-{y}-initial_interval.npy'
                        transition_intervals = f'{model_path}/high-st-{mc}-comp-ref-{x}-{y}-interval.npy'
                    elif method == 'refsplit':
                        initial_distribution = f'{model_path}/high-st-{mc}-comp-refsplit-{x}-{y}-initial_interval.npy'
                        transition_intervals = f'{model_path}/high-st-{mc}-comp-refsplit-{x}-{y}-interval.npy'
                else:
                    if method == 'noref':
                        initial_distribution = f'{model_path}/{mc}-comp-noref-{x}-{y}-initial_interval.npy'
                        transition_intervals = f'{model_path}/{mc}-comp-noref-{x}-{y}-interval.npy'
                    elif method == 'ref':
                        initial_distribution = f'{model_path}/{mc}-comp-ref-{x}-{y}-initial_interval.npy'
                        transition_intervals = f'{model_path}/{mc}-comp-ref-{x}-{y}-interval.npy'
                    elif method == 'refsplit':
                        initial_distribution = f'{model_path}/{mc}-comp-refsplit-{x}-{y}-initial_interval.npy'
                        transition_intervals = f'{model_path}/{mc}-comp-refsplit-{x}-{y}-interval.npy'


            args.trans_path = transition_intervals
            args.init_path = initial_distribution
            args.dump = None
            args.exact = True
            args.precision = 1e-6

            transition_intervals, initial_distribution = load_imc(args)

            # Build the premise monitor on the learned model
            mon, mon_comps = create_monitor(
                transition_intervals,
                initial_distribution,
                "min", 
                True,
                horizon,
                args.dump,
                args.verbose,
                use_exact=args.exact,
                precision=args.precision,
            )

            imc_risks[f'{x}-{y}'] = []

            for t in testing_samples: 
                sub_trace: Trace = t[:initial_amount]
                risk = test_monitor(
                    mon,
                    [sub_trace],
                    obs_func=lambda x: mon_comps.observation_map[x],
                    skip_initial=True,
                    with_tqdm=False,
                    )[sub_trace]
                
               
                imc_risks[f'{x}-{y}'].append(float(risk))

    return imc_risks, imc_transition_counts, stopping_threashold, imc_distances


def aggregated_stats_regression(high_st, coarse, mc, model_path, stats_path, testing_samples, horizon, initial_amount): 

    regression_risks = {}
    regression_ys = {}

    for x in range(1,11):
    #for x in range(5,7):
        if coarse: 
            if high_st:
                paths = glob.glob(f'{model_path}/high-st-{mc}-coarse-comp-reg-{x}_*.npy')
            else:
                paths = glob.glob(f'{model_path}/{mc}-coarse-comp-reg-{x}_*.npy')
        else: 
            if high_st:
                paths = glob.glob(f'{model_path}/high-st-{mc}-comp-reg-{x}_*.npy')
            else:
                paths = glob.glob(f'{model_path}/{mc}-comp-reg-{x}_*.npy')
        
        regression_ys[str(x)] = []

        for path in paths:
            match = re.search(r'_(\d+)\.npy$', path)
            if match:
                regression_ys[str(x)].append(int(match.group(1)))

        for key in regression_ys.keys(): 
            regression_ys[key].sort()

        for y in regression_ys[str(x)]:
            regression_risks[f'{x}-{y}'] = []

            if coarse:
                if high_st: 
                    reg_model = np.load(f'{model_path}/high-st-{mc}-coarse-comp-reg-{x}_{y}.npy', allow_pickle=True).item()
                    observations = np.load(f'{stats_path}/high-st-{mc}-coarse-comp-reg-stats-{x}.npy', allow_pickle=True).item()["observations"]
                else:
                    reg_model = np.load(f'{model_path}/{mc}-coarse-comp-reg-{x}_{y}.npy', allow_pickle=True).item()
                    observations = np.load(f'{stats_path}/{mc}-coarse-comp-reg-stats-{x}.npy', allow_pickle=True).item()["observations"]
            else: 
                if high_st: 
                    reg_model = np.load(f'{model_path}/high-st-{mc}-comp-reg-{x}_{y}.npy', allow_pickle=True).item()
                    observations = np.load(f'{stats_path}/high-st-{mc}-comp-reg-stats-{x}.npy', allow_pickle=True).item()["observations"]
                else: 
                    reg_model = np.load(f'{model_path}/{mc}-comp-reg-{x}_{y}.npy', allow_pickle=True).item()
                    observations = np.load(f'{stats_path}/{mc}-comp-reg-stats-{x}.npy', allow_pickle=True).item()["observations"]

            column_names = [f"Step{s}_Obs{o}" for s in range(initial_amount) for o in observations]

            for t in testing_samples: 
                sub_trace: Trace = t[:initial_amount]
                reg_sub_trace = prep_trace_for_regression(sub_trace, observations)
                X = pd.DataFrame([reg_sub_trace], columns=column_names)
                prob = reg_model.predict_proba(X)    
                regression_risks[f'{x}-{y}'].append(float(prob[:, 1].item()))

    return regression_risks, regression_ys



def aggreagted_stats_conformal(high_st, new_noisy, coarse, mc, model_path, stats_path):

    conformal_risks = {}
    conformal_ys = {}

    for x in range(8,9):
        if coarse: 
            if high_st:
                paths = glob.glob(f'{model_path}/high-st-{mc}_coarse_comp_conformal_pred_state_estimator_{x}_*.pt')
            else: 
                paths = glob.glob(f'{model_path}/{mc}_coarse_comp_conformal_pred_state_estimator_{x}_*.pt')
        else: 
            if high_st:
                paths = glob.glob(f'{model_path}/high-st-{mc}_comp_conformal_pred_state_estimator_{x}_*.pt')
            else:
                paths = glob.glob(f'{model_path}/{mc}_comp_conformal_pred_state_estimator_{x}_*.pt')

        conformal_ys[str(x)] = []

        for path in paths: 
            match = re.search(r'_(\d+)\.pt$', path)
            if match: 
                conformal_ys[str(x)].append(int(match.group(1)))
        
        for key in conformal_ys.keys(): 
            conformal_ys[key].sort()

        for y in conformal_ys[str(x)]:
            conformal_risks[f'{x}-{y}'] = [] 

            if coarse: 
                if high_st:
                    state_estimator = torch.load(f'{model_path}/high-st-{mc}_coarse_comp_conformal_pred_state_estimator_{x}_{y}.pt', weights_only=False)
                    label_estimator = torch.load(f'{model_path}/high-st-{mc}_coarse_comp_conformal_pred_label_estimator_{x}_{y}.pt', weights_only=False)
                    cp_classification = torch.load(f'{model_path}/high-st-{mc}_coarse_comp_conformal_pred_cp_classification_{x}_{y}.pt', weights_only=False)

                    with open(f'{stats_path}/high-st-{mc}_coarse_comp_conformal_pred_rejection_classifier_{x}_{y}.pickle', 'rb') as f:
                        rej_classifier = pickle.load(f)
            
                    rejection_classifier = rej_classifier['rej_rule']

                    with open(f'{stats_path}/high-st-{mc}_coarse_comp_conformal_pred_conformal_stats_{x}_{y}.pickle', 'rb') as f:
                        conformal_stats = pickle.load(f)
                else:
                    state_estimator = torch.load(f'{model_path}/{mc}_coarse_comp_conformal_pred_state_estimator_{x}_{y}.pt', weights_only=False)
                    label_estimator = torch.load(f'{model_path}/{mc}_coarse_comp_conformal_pred_label_estimator_{x}_{y}.pt', weights_only=False)
                    cp_classification = torch.load(f'{model_path}/{mc}_coarse_comp_conformal_pred_cp_classification_{x}_{y}.pt', weights_only=False)

                    with open(f'{stats_path}/{mc}_coarse_comp_conformal_pred_rejection_classifier_{x}_{y}.pickle', 'rb') as f:
                        rej_classifier = pickle.load(f)
            
                    rejection_classifier = rej_classifier['rej_rule']

                    with open(f'{stats_path}/{mc}_coarse_comp_conformal_pred_conformal_stats_{x}_{y}.pickle', 'rb') as f:
                        conformal_stats = pickle.load(f)
                
            else: 
                if high_st:
                    state_estimator = torch.load(f'{model_path}/high-st-{mc}_comp_conformal_pred_state_estimator_{x}_{y}.pt', weights_only=False)
                    label_estimator = torch.load(f'{model_path}/high-st-{mc}_comp_conformal_pred_label_estimator_{x}_{y}.pt', weights_only=False)
                    cp_classification = torch.load(f'{model_path}/high-st-{mc}_comp_conformal_pred_cp_classification_{x}_{y}.pt', weights_only=False)

                    with open(f'{stats_path}/high-st-{mc}_comp_conformal_pred_rejection_classifier_{x}_{y}.pickle', 'rb') as f:
                        rej_classifier = pickle.load(f)
            
                    rejection_classifier = rej_classifier['rej_rule']

                    with open(f'{stats_path}/high-st-{mc}_comp_conformal_pred_conformal_stats_{x}_{y}.pickle', 'rb') as f:
                        conformal_stats = pickle.load(f)
                else:
                    state_estimator = torch.load(f'{model_path}/{mc}_comp_conformal_pred_state_estimator_{x}_{y}.pt', weights_only=False)
                    label_estimator = torch.load(f'{model_path}/{mc}_comp_conformal_pred_label_estimator_{x}_{y}.pt', weights_only=False)
                    cp_classification = torch.load(f'{model_path}/{mc}_comp_conformal_pred_cp_classification_{x}_{y}.pt', weights_only=False)

                    with open(f'{stats_path}/{mc}_comp_conformal_pred_rejection_classifier_{x}_{y}.pickle', 'rb') as f:
                        rej_classifier = pickle.load(f)
            
                    rejection_classifier = rej_classifier['rej_rule']

                    with open(f'{stats_path}/{mc}_comp_conformal_pred_conformal_stats_{x}_{y}.pickle', 'rb') as f:
                        conformal_stats = pickle.load(f)


            new_noisy_scaled = -1+2*(new_noisy - conformal_stats['dataset.MIN[1]'])/(conformal_stats['dataset.MAX[1]']-conformal_stats['dataset.MIN[1]'])
            Y1 = np.transpose(new_noisy_scaled, (0,2,1))
            Y1t = Variable(FloatTensor(Y1))

            state_estimator.eval()    
            state_estim = state_estimator(Y1t)
            label_estimator.eval()
            label_hypothesis = label_estimator(state_estim)
    
            label_prob = torch.nn.functional.softmax(label_hypothesis, dim=1)
            error_prob = label_prob[:, 1]
            error_prob = error_prob.tolist()


            pool_conf_cred = cp_classification.compute_confidence_credibility(np.transpose(new_noisy_scaled,(0,2,1)))
            keep_mask = utils.apply_svc_query_strategy(rejection_classifier, pool_conf_cred)

            for u in range(len(error_prob)): 
                if keep_mask[u] == -1.0: 
                    error_prob[u] = 1.0 
            
            for u in range(len(error_prob)):
                conformal_risks[f'{x}-{y}'].append(error_prob[u])


    return conformal_risks, conformal_ys

def find_first_triplet_below_threshold(distances, threshold):
    for i in range(2, len(distances)):
        if distances[i] < threshold and distances[i-1] < threshold and distances[i-2] < threshold:
            return i 


def roc_curve_per_threashold(coarse, current_threashold, alarms, imc_risks, imc_risks_ref, target_risks, imc_distances, imc_distances_ref, imc_risks_ref_splitting, imc_distances_ref_splitting):

    #REFINEMENT WITH SPLITTING

    index_ref_splitting = {}

    #for x in range(1,11):
    for x in range(5,7):
        index_ref_splitting[str(x)] =  (find_first_triplet_below_threshold(imc_distances_ref_splitting[str(x)],current_threashold) +1)
    
    imc_risks_threashold_ref_split = {}

    for key in imc_risks_ref_splitting.keys():
        for i in index_ref_splitting.keys(): 
            if i == key.split('-')[0]:
                if key.split('-')[1] == str(index_ref_splitting[i]):
                    imc_risks_threashold_ref_split[i] = imc_risks_ref_splitting[key]

    #REFINEMENT

    index_ref = {}

    for x in range(1,11):
    #for x in range(5,7):
        index_ref[str(x)] =  (find_first_triplet_below_threshold(imc_distances_ref[str(x)],current_threashold) +1)
    
    imc_risks_threashold_ref = {}

    for key in imc_risks_ref.keys():
        for i in index_ref.keys(): 
            if i == key.split('-')[0]:
                if key.split('-')[1] == str(index_ref[i]):
                    imc_risks_threashold_ref[i] = imc_risks_ref[key]
    
    #NO REFINEMENT

    index = {}

    for x in range(1,11):
    #for x in range(5,7):
        index[str(x)] =  (find_first_triplet_below_threshold(imc_distances[str(x)],current_threashold) +1)
    
    imc_risks_threashold = {}

    for key in imc_risks.keys():
        for i in index.keys(): 
            if i == key.split('-')[0]:
                if key.split('-')[1] == str(index[i]):
                    imc_risks_threashold[i] = imc_risks[key]
    
    plt.figure()
    fig, ax = plt.subplots(figsize=(16, 12))



    #REFINEMENT WITH SPLITTING MEAN PERFORMANCE
    ref_splitting_imc_roc_data_mean = {}
    
    for key in imc_risks_threashold_ref_split.keys(): 

        fpr, tpr, thresholds = metrics.roc_curve(alarms, imc_risks_threashold_ref_split[key])
        roc_auc = metrics.auc(fpr, tpr)
        ref_splitting_imc_roc_data_mean[key] = [fpr, tpr, roc_auc]

    mean_fpr = np.linspace(0, 1, 100)

    tprs = []
    aucs = []

    for key in ref_splitting_imc_roc_data_mean.keys(): 
        interp_tpr = np.interp(mean_fpr, ref_splitting_imc_roc_data_mean[key][0], ref_splitting_imc_roc_data_mean[key][1])
        aucs.append(ref_splitting_imc_roc_data_mean[key][2])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  
    mean_auc = np.mean(aucs)

    plt.plot(mean_fpr, mean_tpr, color = 'aqua',  label = f'Refinement with splitting, (Mean AUC = {mean_auc:.2f})', linewidth=5, linestyle= '-.')

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


    #REFINEMENT MEAN PERFORMANCE
    ref_imc_roc_data_mean = {}
    
    for key in imc_risks_threashold_ref.keys(): 

        fpr, tpr, thresholds = metrics.roc_curve(alarms, imc_risks_threashold_ref[key])
        roc_auc = metrics.auc(fpr, tpr)
        ref_imc_roc_data_mean[key] = [fpr, tpr, roc_auc]

    mean_fpr = np.linspace(0, 1, 100)
 

    tprs = []
    aucs = []

    for key in ref_imc_roc_data_mean.keys(): 
        interp_tpr = np.interp(mean_fpr, ref_imc_roc_data_mean[key][0], ref_imc_roc_data_mean[key][1])
        aucs.append(ref_imc_roc_data_mean[key][2])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

     
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  
    mean_auc = np.mean(aucs)

    plt.plot(mean_fpr, mean_tpr, color = 'blue',  label = f'Refinement, (Mean AUC = {mean_auc:.2f})', linewidth=5, linestyle= ':')

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


    #NO REFINEMENT MEAN PERFORMANCE
    imc_roc_data_mean = {}
    
    for key in imc_risks_threashold.keys(): 
        fpr, tpr, thresholds = metrics.roc_curve(alarms, imc_risks_threashold[key])
        roc_auc = metrics.auc(fpr, tpr)
        imc_roc_data_mean[key] = [fpr, tpr, roc_auc]

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for key in imc_roc_data_mean.keys(): 
        interp_tpr = np.interp(mean_fpr, imc_roc_data_mean[key][0], imc_roc_data_mean[key][1])
        aucs.append(imc_roc_data_mean[key][2])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  
    mean_auc = np.mean(aucs)


    plt.plot(mean_fpr, mean_tpr, color = 'red',  label = f'No Refinement, (Mean AUC = {mean_auc:.2f})', linewidth=5, linestyle= '--')

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

    ax.plot(fpr, tpr, color = 'black', label = f'Target Monitor, AUC = {target_auc:.2f}', linewidth=5, linestyle='-', marker='D')

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
        plt.title(f'{args.mc} coarse, stopping threashold: {current_threashold}', fontsize=35)
    else: 
        plt.title(f'{args.mc}, stopping threashold: {current_threashold}', fontsize=35)


    if coarse:
        plt.savefig(f"/workspaces/premise/premise/analysis/rq2_{args.mc}_coarse_ROC_threashold_comparison_{current_threashold}.pdf", dpi=300,  bbox_inches='tight')
    else:
        plt.savefig(f"/workspaces/premise/premise/analysis/rq2_{args.mc}_ROC_threashold_comparison_{current_threashold}.pdf", dpi=300,  bbox_inches='tight')
    plt.show()


#def plot_roc_curve(coarse, alarms, imc_risks_ref, regression_risks, regression_ys, conformal_risks, conformal_ys, target_risks):
def plot_roc_curve(high_st, coarse, alarms, imc_risks_ref, regression_risks, regression_ys, target_risks):  

    imc_ref_final_risks = {}

    ys = []
    for key in imc_risks_ref.keys():
        ys.append(int(key.split('-')[1]))
    
    for key in imc_risks_ref.keys():
        if key.split('-')[1] == str(max(ys)):
            imc_ref_final_risks[key.split('-')[0]] = imc_risks_ref[key]

    

    imc_ref_split_final_risks = {}  

    ys_splitting = []
    for key in imc_risks_ref_splitting.keys():
        ys_splitting(int(key.split('-')[1]))
    
    for key in imc_risks_ref_splitting.keys():
        if key.split('-')[1] == str(max(ys_splitting)):
            imc_ref_split_final_risks[key.split('-')[0]] = imc_risks_ref_splitting[key]



    reg_final_risks = {}

    for x in range(1,11):
    #for x in range(5,7): 
        for key in regression_risks.keys(): 
            if int(key.split('-')[1]) == max(regression_ys[str(x)]): 
                reg_final_risks[str(x)] = regression_risks[key]
    
    
    conformal_final_risks = {}

    for x in range(8,9): 
        for key in conformal_risks.keys(): 
            if int(key.split('-')[1]) == max(conformal_ys[str(x)]): 
                conformal_final_risks[str(x)] = conformal_risks[key]




    plt.figure()
    fig, ax = plt.subplots(figsize=(16, 12))
    
    #REFINEMENT MEAN PERFORMANCE
    ref_imc_roc_data = {}
    
    for key in imc_ref_final_risks.keys(): 

        fpr, tpr, thresholds = metrics.roc_curve(alarms, imc_ref_final_risks[key])
        roc_auc = metrics.auc(fpr, tpr)
        ref_imc_roc_data[key] = [fpr, tpr, roc_auc]

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for key in ref_imc_roc_data.keys(): 
        interp_tpr = np.interp(mean_fpr, ref_imc_roc_data[key][0], ref_imc_roc_data[key][1])
        aucs.append(ref_imc_roc_data[key][2])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  
    mean_auc = np.mean(aucs)

    plt.plot(mean_fpr, mean_tpr, color = 'blue',  label = f'Refinement, (Mean AUC = {mean_auc:.2f})', linewidth=5, linestyle= ':')

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

    #REFINEMENT WITH SPLITTING MEAN PERFORMANCE
    ref_splitting_imc_roc_data = {}
    
    for key in imc_ref_split_final_risks.keys(): 

        fpr, tpr, thresholds = metrics.roc_curve(alarms, imc_ref_split_final_risks[key])
        roc_auc = metrics.auc(fpr, tpr)
        ref_splitting_imc_roc_data[key] = [fpr, tpr, roc_auc]

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for key in ref_splitting_imc_roc_data.keys(): 
        interp_tpr = np.interp(mean_fpr, ref_splitting_imc_roc_data[key][0], ref_splitting_imc_roc_data[key][1])
        aucs.append(ref_splitting_imc_roc_data[key][2])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  
    mean_auc = np.mean(aucs)

    plt.plot(mean_fpr, mean_tpr, color = 'aqua',  label = f'Refinement with splitting, (Mean AUC = {mean_auc:.2f})', linewidth=5, linestyle= '-.')

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

    #REGRESSION MEAN PERFORMANCE
    regression_roc_data = {}
    
    for key in reg_final_risks.keys(): 

        fpr, tpr, thresholds = metrics.roc_curve(alarms, reg_final_risks[key])
        roc_auc = metrics.auc(fpr, tpr)
        regression_roc_data[key] = [fpr, tpr, roc_auc]

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for key in regression_roc_data.keys(): 
        interp_tpr = np.interp(mean_fpr, regression_roc_data[key][0], regression_roc_data[key][1])
        aucs.append(regression_roc_data[key][2])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  
    mean_auc = np.mean(aucs)

    plt.plot(mean_fpr, mean_tpr, color = 'green',  label = f'Regression, (Mean AUC = {mean_auc:.2f})', linewidth=5, linestyle='--')

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

    #CONFORMAL MEAN PERFORMANCE 

    conformal_roc_data = {}
    
    for key in conformal_final_risks.keys(): 

        fpr, tpr, thresholds = metrics.roc_curve(alarms, conformal_final_risks[key])
        roc_auc = metrics.auc(fpr, tpr)
        conformal_roc_data[key] = [fpr, tpr, roc_auc]

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for key in conformal_roc_data.keys(): 
        interp_tpr = np.interp(mean_fpr, conformal_roc_data[key][0], conformal_roc_data[key][1])
        aucs.append(conformal_roc_data[key][2])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  
    mean_auc = np.mean(aucs)

    plt.plot(mean_fpr, mean_tpr, color = 'orange',  label = f'Conformal Prediction, (Mean AUC = {mean_auc:.2f})', linewidth=5)

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

    ax.plot(fpr, tpr, color = 'black', label = f'Target Monitor, AUC = {target_auc:.2f}', linewidth=5, linestyle='-', marker='D')

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

    if coarse:
        if high_st:
            plt.title(f'{args.mc} coarse high st', fontsize=35)
        else: 
            plt.title(f'{args.mc} coarse', fontsize=35)
    else: 
        if high_st:
            plt.title(f'{args.mc} high st', fontsize=35)
        else:
            plt.title(f'{args.mc}', fontsize=35)
    plt.tight_layout()

    if coarse:
        if high_st:
            plt.savefig(f"/workspaces/premise/premise/analysis/rq_3_high-st-{args.mc}_coarse_model_based_model_free_ROC.pdf", dpi=300,  bbox_inches='tight')
        else:
            plt.savefig(f"/workspaces/premise/premise/analysis/rq_3_{args.mc}_coarse_model_based_model_free_ROC.pdf", dpi=300,  bbox_inches='tight')
    else: 
        if high_st:
            plt.savefig(f"/workspaces/premise/premise/analysis/rq_3_high-st-{args.mc}_model_based_model_free_ROC.pdf", dpi=300,  bbox_inches='tight')
        else:
            plt.savefig(f"/workspaces/premise/premise/analysis/rq_3_{args.mc}_model_based_model_free_ROC.pdf", dpi=300,  bbox_inches='tight')

    plt.show()



def auc_graph_prep(imc_risks, target_risks, alarms, imc_risks_ref, imc_transition_counts_ref, imc_risks_ref_splitting, imc_transition_counts_ref_splitting): 

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
        fpr, tpr, threshold = metrics.roc_curve(alarms, imc_risks_ref_splitting[imc_ref_key_splitting])
        roc_auc = metrics.auc(fpr, tpr)
        imc_ref_splitting_auc[imc_ref_key_splitting] = roc_auc

    
    """ reg_auc = {}

    for reg_key in regression_risks.keys(): 
        fpr, tpr, threshold = metrics.roc_curve(alarms, regression_risks[reg_key])
        roc_auc = metrics.auc(fpr, tpr)
        reg_auc[reg_key] = roc_auc
 """
    """ 
    conformal_auc = {}

    for conformal_key in conformal_risks.keys():
        fpr, tpr, threshold = metrics.roc_curve(alarms, conformal_risks[conformal_key])
        roc_auc = metrics.auc(fpr, tpr)
        conformal_auc[conformal_key] = roc_auc """


    imc_results = {}

    for x in range(1,11):
    #for x in range(5,7):
        imc_results[str(x)] = []

    for key in imc_auc.keys():
        for entry in imc_results.keys():
            x = key.split('-')[0]
            if x == entry:
                imc_results[x].append(imc_auc[key])
 

    imc_ref_results = {}

    for x in range(1,11):
    #for x in range(5,7):
        imc_ref_results[str(x)] = []

    for key in imc_ref_auc.keys():
        for entry in imc_ref_results.keys():
            x = key.split('-')[0]
            if x == entry:
                imc_ref_results[x].append(imc_ref_auc[key])


    imc_ref_splitting_results = {}

    for x in range(1,11):
    #for x in range(5,7):
        imc_ref_splitting_results[str(x)] = []

    for key in imc_ref_splitting_auc.keys():
        for entry in imc_ref_splitting_results.keys():
            x = key.split('-')[0]
            if x == entry:
                imc_ref_splitting_results[x].append(imc_ref_splitting_auc[key])

    
    """ reg_results = {}

    for x in range(1,11):
        reg_results[str(x)] = []


    for key in reg_auc.keys():
        for entry in reg_results.keys():
            x = key.split('-')[0]
            if x == entry:
                reg_results[x].append(reg_auc[key]) """

    """ 
    conformal_results = {}

    for x in range(8,9): 
        conformal_results[str(x)] = []

    for key in conformal_auc.keys(): 
        for entry in conformal_results.keys():
            x = key.split('-')[0]
            if x == entry: 
                conformal_results[x].append(conformal_auc[key]) """


    return target_auc, imc_results, imc_ref_results, imc_transition_counts_ref,  imc_ref_splitting_results, imc_transition_counts_ref_splitting


def plotting(coarse, target_auc, imc_results, imc_transition_counts, imc_ref_results, imc_transition_counts_ref, horizon, initial_amount, imc_ref_splitting_results, imc_transition_counts_ref_splitting): 

    RS_sets = []
    R_sets = []
    NR_sets = []

    #REG_sets = []
    #CONF_sets = []

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

    
    """ for key in reg_results.keys():
        total_state_count = []
  
        for val in regression_ys[key]:
            total_state_count.append(val*(horizon+initial_amount))
        
        REG_sets.append((total_state_count, reg_results[key]))
 """
    """
    for key in conformal_results.keys():
        total_state_count = []

        for val in conformal_ys[key]:
            total_state_count.append(val*(horizon+initial_amount))

        CONF_sets.append((total_state_count, conformal_results[key])) 
    """        

    #log = True
    log = False

    #REFINEMENT AVERAGE PERFORMANCE
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
    ax.plot(x_values, y, color = 'black', label = 'Target Monitor', linewidth=5, linestyle='-', marker='D')

    #Plot mean line
    ax.plot(
        x_values,
        mean_auc,
        color='blue',
        label = 'Refinement',
        linewidth=5,
        linestyle=':',
        )

    #Add shaded area for spread
    ax.fill_between(
            x_values,
            mean_auc - std_auc,
            mean_auc + std_auc,
            alpha=0.2,
            color='blue',
        )

    #REFINEMENT WITH SPLITTING AVERAGE PERFORMANCE
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
 

    #Plot mean line
    ax.plot(
        x_values,
        mean_auc,
        color='aqua',
        label = 'Refinement with splitting',
        linewidth=5,
        linestyle='-.',
        )

    #Add shaded area for spread
    ax.fill_between(
            x_values,
            mean_auc - std_auc,
            mean_auc + std_auc,
            alpha=0.2,
            color='aqua',
        )

    #NO REFINEMENT AVERAGE PERFORMANCE

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


    #Plot mean line
    ax.plot(
        N_x_values,
        N_mean_auc,
        color='red',
        label = 'No refinement',
        linestyle='--',
        linewidth=5,
        )

    #Add shaded area for spread
    ax.fill_between(
            N_x_values,
            N_mean_auc - N_std_auc,
            N_mean_auc + N_std_auc,
            alpha=0.2,
            color='red',
        )

    """ #REGRESSION MODEL AVERAGE PERFORMANCE

    REG_transitions_data = []
    REG_auc_data = []

    for entry in REG_sets:
        transitions = entry[0]
        auc_daum = entry[1]

        REG_transitions_data.append(transitions)
        REG_auc_data.append(auc_daum)

    # Find common x range for interpolation
    REG_min_x = max(min(transitions) for transitions in REG_transitions_data)
    REG_max_x = min(max(transitions) for transitions in REG_transitions_data)
    REG_x_values = np.linspace(REG_min_x, REG_max_x, 500)

    # Interpolate all runs to common x values
    REG_interpolated_auc = []
    for auc, transitions in zip(REG_auc_data, REG_transitions_data):
        if log:
            auc = np.log10(auc)
        REG_interpolated = np.interp(REG_x_values, transitions, auc)
        if log:
            REG_interpolated = np.power(10, REG_interpolated)
        REG_interpolated_auc.append(REG_interpolated)

    # Calculate mean and std for interpolated y values
    REG_auc_array = np.array(REG_interpolated_auc)
    REG_mean_auc = np.mean(REG_auc_array, axis=0)
    REG_std_auc = np.std(REG_auc_array, axis=0)
    REG_min_auc = np.min(REG_auc_array, axis=0)
    REG_max_auc = np.max(REG_auc_array, axis=0)

    #Plot mean line
    ax.plot(
        REG_x_values,
        REG_mean_auc,
        color='green',
        label = 'Regression',
        linestyle='-.',
        linewidth=5,
        )

    #Add shaded area for spread
    ax.fill_between(
            REG_x_values,
            REG_mean_auc - REG_std_auc,
            REG_mean_auc + REG_std_auc,
            alpha=0.2,
            color='green'
        ) """
    
    """ 
    #CONFORMAL PREDICTION MODEL AVERAGE PERFORMANCE

    CONF_transitions_data = []
    CONF_auc_data = []

    for entry in CONF_sets:
        transitions = entry[0]
        auc_daum = entry[1]

        CONF_transitions_data.append(transitions)
        CONF_auc_data.append(auc_daum)

    # Find common x range for interpolation
    CONF_min_x = max(min(transitions) for transitions in CONF_transitions_data)
    CONF_max_x = min(max(transitions) for transitions in CONF_transitions_data)
    CONF_x_values = np.linspace(CONF_min_x, CONF_max_x, 500)

    # Interpolate all runs to common x values
    CONF_interpolated_auc = []
    for auc, transitions in zip(CONF_auc_data, CONF_transitions_data):
        if log:
            auc = np.log10(auc)
        CONF_interpolated = np.interp(CONF_x_values, transitions, auc)
        if log:
            CONF_interpolated = np.power(10, CONF_interpolated)
        CONF_interpolated_auc.append(CONF_interpolated)

    # Calculate mean and std for interpolated y values
    CONF_auc_array = np.array(CONF_interpolated_auc)
    CONF_mean_auc = np.mean(CONF_auc_array, axis=0)
    CONF_std_auc = np.std(CONF_auc_array, axis=0)
    CONF_min_auc = np.min(CONF_auc_array, axis=0)
    CONF_max_auc = np.max(CONF_auc_array, axis=0)

    #Plot mean line
    ax.plot(
        CONF_x_values,
        CONF_mean_auc,
        color='orange',
        label = 'Conformal prediction',
        linewidth=5,
        )

    #Add shaded area for spread
    ax.fill_between(
            CONF_x_values,
            CONF_mean_auc - CONF_std_auc,
            CONF_mean_auc + CONF_std_auc,
            alpha=0.2,
            color='yellow',
        ) 
 """
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((4, 4))  # Force 10^4 scale
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='both', labelsize=30)
    ax.xaxis.get_offset_text().set_size(30)


    ax.set_xlabel("State count", fontsize=35)
    ax.set_ylabel("AUC", fontsize=35)
    ax.legend(loc="lower right", fontsize = 30)
    if log:
        plt.yscale("log")
    else:
        plt.ylim(bottom=0)
    ax.grid(True)
    plt.subplots_adjust(bottom=0.25)
    if coarse: 
        plt.title(f'{args.mc} coarse', fontsize=35)
    else:
        plt.title(f'{args.mc}', fontsize=35) 
    plt.tight_layout()
    if coarse:
        plt.savefig(f"/workspaces/premise/premise/analysis/rq_2_{args.mc}_coarse_AUC_ref_no_ref.pdf", dpi=300, bbox_inches='tight')
    else: 
        plt.savefig(f"/workspaces/premise/premise/analysis/rq_2_{args.mc}_AUC_ref_no_ref.pdf", dpi=300, bbox_inches='tight')

    plt.show()


def main_imc(args: argparse.Namespace):

    suo, initial_amount, horizon = build_suo(args)

    length = initial_amount + horizon

    testing_samples = []
    for x in range(args.testing_samples):
            path = suo.generate_random_traces([], length)[0]
            testing_samples.append(tuple(path))


    models_dict = {"IP": InvertedPendulum(), "MC": mc_model(horizon)}
    model = models_dict['MC']
    model = mc_model(horizon)

    noisy_measurements = model.get_noisy_measurments(testing_samples, horizon)

    alarms = aggregted_alarms(testing_samples)
    target_risks = stats_true(horizon, initial_amount, testing_samples, suo)

    coarse = args.coarse
    mc = args.mc
    model_path = args.model_path
    stats_path = args.stats_path


    imc_risks_ref_splitting, imc_transition_counts_ref_splitting, imc_stopping_threashold_ref_splitting, imc_distances_ref_splitting = aggregated_stats_imc(args.high_st, coarse, 'refsplit', mc, model_path, stats_path, initial_amount, horizon, args, testing_samples)
    imc_risks, imc_transition_counts, imc_stopping_threashold, imc_distances = aggregated_stats_imc(args.high_st, coarse, 'noref', mc, model_path, stats_path, initial_amount, horizon, args, testing_samples)
    imc_risks_ref, imc_transition_counts_ref, imc_stopping_threashold_ref, imc_distances_ref = aggregated_stats_imc(args.high_st, coarse, 'ref', mc, model_path, stats_path, initial_amount, horizon, args, testing_samples)

    regression_risks, regression_ys = aggregated_stats_regression(args.high_st, coarse, mc, model_path, stats_path, testing_samples, horizon, initial_amount)
    conformal_risks, conformal_ys = aggreagted_stats_conformal(args.high_st, new_noisy, coarse, mc, model_path, stats_path)      
            
    target_auc, imc_results, imc_ref_results, imc_transition_counts_ref,  imc_ref_splitting_results, imc_transition_counts_ref_splitting = auc_graph_prep(imc_risks, target_risks, alarms, imc_risks_ref, imc_transition_counts_ref, imc_risks_ref_splitting, imc_transition_counts_ref_splitting)
    plotting(coarse, target_auc, imc_results, imc_transition_counts, imc_ref_results, imc_transition_counts_ref, horizon, initial_amount, imc_ref_splitting_results, imc_transition_counts_ref_splitting)
    
    plot_roc_curve(args.high_st, coarse, alarms, imc_risks_ref, regression_risks, regression_ys, target_risks)

    roc_curve_per_threashold(coarse, 0.2, alarms, imc_risks, imc_risks_ref, target_risks, imc_distances, imc_distances_ref, imc_risks_ref_splitting, imc_distances_ref_splitting)
    roc_curve_per_threashold(coarse, 0.1, alarms, imc_risks, imc_risks_ref, target_risks, imc_distances, imc_distances_ref, imc_risks_ref_splitting, imc_distances_ref_splitting)
    roc_curve_per_threashold(coarse, 0.05, alarms, imc_risks, imc_risks_ref, target_risks, imc_distances, imc_distances_ref, imc_risks_ref_splitting, imc_distances_ref_splitting)
    roc_curve_per_threashold(coarse, 0.025, alarms, imc_risks, imc_risks_ref, target_risks, imc_distances, imc_distances_ref, imc_risks_ref_splitting, imc_distances_ref_splitting)
    roc_curve_per_threashold(coarse, 0.015, alarms, imc_risks, imc_risks_ref, target_risks, imc_distances, imc_distances_ref, imc_risks_ref_splitting, imc_distances_ref_splitting)
    roc_curve_per_threashold(coarse, 0.01, alarms, imc_risks, imc_risks_ref, target_risks, imc_distances, imc_distances_ref, imc_risks_ref_splitting, imc_distances_ref_splitting)
    roc_curve_per_threashold(coarse, 0.005, alarms, imc_risks, imc_risks_ref, target_risks, imc_distances, imc_distances_ref, imc_risks_ref_splitting, imc_distances_ref_splitting)
    
    #plot_roc_curve(coarse, alarms, imc_risks_ref, regression_risks, regression_ys, conformal_risks, conformal_ys, target_risks)
    

def build_learning_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Learning Parameters")

    group.add_argument("--model_name", type=str, default="MC", help="Name of the conformal prediction model (first letters code).")
    group.add_argument("-s", "--testing_samples", type=int, default = 200, help="Total number of samples used in learning")
    group.add_argument("--no-target", action="store_true", default=False, help="Do not use the target monitor" )


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

    parser.add_argument('--model_path', 
                        type = str, 
                        help = 'Path to models',
    )

    parser.add_argument('--stats_path', 
                        type = str, 
                        help ='Path to stats',
    )

    parser.add_argument("--coarse",
                        type = bool, 
                        default= False, 
                        help = "If the leared model is coarse or not")
    
    parser.add_argument("--high_st", 
                        type = bool, 
                        default = False, 
                        help = "If higher stopping threashold is used")

    return parser


if __name__ == "__main__":
    parser = testing_argsparser()
    args = parser.parse_args()
    main_imc(args)


# python -m premise.interval.rq_3 --mc airportA-7-10-10 --model_path /workspaces/premise/out/models/2025-07-17 --stats_path /workspaces/premise/out/stats/2025-07-17 --coarse 
# python -m premise.interval.rq_3 --mc evadeV-5-3 --model_path /workspaces/premise/out/models/2025-07-17 --stats_path /workspaces/premise/out/stats/2025-07-17


#coarse = True 

    #imc_model {model_path}/{mc}-coarse-comp-noref
    #imc_stats {stats_path}/{mc}-coarse_norefinement-stats
    #imc_model_ref {model_path}/{mc}-coarse-comp-ref
    #imc_stats_ref {stats_path}/{mc}-coarse_refinement-stats
    #imc_model_ref_splitting {model_path}/{mc}-coarse-comp-refsplit
    #imc_stats_ref_splitting {stats_path}/{mc}-coarse_refsplitinement-stats

    #regression_model {model_path}/{mc}-coarse-comp-reg
    #regression_stats {stats_path}/{mc}-coarse-comp-reg-stats

    #se_path {model_path}/{mc}_coarse_comp_conformal_pred_state_estimator
    #error_path {model_path}/{mc}_coarse_comp_conformal_pred_label_estimator
    #rej_path {stats_path}/{mc}_coarse_comp_conformal_pred_label_estimator 
    #stats_path {stats_path}/{mc}_coarse_comp_conformal_pred_rejection_classifier
    #cp_classification_path {model_path}/{mc}_coarse_comp_conformal_pred_cp_classification


#coarse = False 

    #imc_model {model_path}/{mc}-comp-noref
    #imc_stats {stats_path}/{mc}-comp-noref-stats
    #imc_model_ref {model_path}/{mc}-comp-ref
    #imc_stats_ref {stats_path}/{mc}-comp-ref-stats
    #imc_model_ref_splitting {model_path}/{mc}-comp-refsplit
    #imc_stats_ref_splitting  {stats_path}/{mc}-comp-refsplit-stats

    #regression_model {model_path}/{mc}-comp-reg
    #regression_stats {stats_path}/{mc}-comp-reg-stats

    #se_path {model_path}/{mc}_comp_conformal_pred_state_estimator
    #error_path {model_path}/{mc}_comp_conformal_pred_label_estimator
    #rej_path {stats_path}/{mc}_comp_conformal_pred_label_estimator 
    #stats_path {stats_path}/{mc}_comp_conformal_pred_rejection_classifier
    #cp_classification_path {model_path}/{mc}_comp_conformal_pred_cp_classification

    
