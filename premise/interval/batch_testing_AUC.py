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
 


def aggregated_stats_imc(path, stats_path, initial_amount, horizon, args, testing_samples): 

    imc_risks = {}

    imc_transition_counts = {}

    for x in range(1,11):

        print(f'Experiment number {x}')
        statistics = np.load(f'{stats_path}-{x}.npy', allow_pickle=True)

        obj = statistics.item()     
        imc_transition_count = obj['transitions_learned']
        imc_transition_counts[str(x)]= imc_transition_count

        for y in range(1, len(imc_transition_count) +1): 
            initial_distribution = f'{path}-{x}-{y}-initial_interval.npy'
            transition_intervals = f'{path}-{x}-{y}-interval.npy'

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

    return imc_risks, imc_transition_counts


def aggregted_alarms(testing_samples): 
    alarms = []

    for trace in tqdm(testing_samples):
            alarms.append(any([s[2] for s in trace]))

    alarms = np.array(alarms).astype(int)
    
    return alarms

def aggregated_stats_regression(regression_model, regression_stats, testing_samples, horizon, initial_amount): 

    regression_risks = {}

    for x in range(1,11):
        paths = glob.glob(f'{regression_model}-{x}_*.npy')

        regression_ys = []
        for path in paths:
            match = re.search(r'_(\d+)\.npy$', path)
            if match:
                regression_ys.append(int(match.group(1)))

        for y in regression_ys:
            regression_risks[f'{x}-{y}'] = []

            reg_model = np.load(f'{regression_model}-{x}_{y}.npy', allow_pickle=True).item()
            observations = np.load(f'{regression_stats}-{x}.npy', allow_pickle=True).item()["observations"]
            column_names = [f"Step{s}_Obs{o}" for s in range(initial_amount) for o in observations]

            for t in testing_samples: 
                sub_trace: Trace = t[:initial_amount]
                reg_sub_trace = prep_trace_for_regression(sub_trace, observations)
                X = pd.DataFrame([reg_sub_trace], columns=column_names)
                prob = reg_model.predict_proba(X)    
                regression_risks[f'{x}-{y}'].append(float(prob[:, 1].item()))


    return regression_risks, regression_ys

def stats_true(horizon, initial_amount, testing_samples, suo): 

    target_risks = []

    target_monitor = suo.create_target_monitor()

    for t in testing_samples: 
        sub_trace: Trace = t[:initial_amount]
        target_risk = test_monitor(
                    target_monitor,
                    [sub_trace],
                    with_tqdm=False,
                )

        target_risk = target_risk[sub_trace]
        print(target_risk)
        target_risks.append(target_risk)   

    return target_risks


def auc_graph_prep(alarms, imc_risks, imc_risks_ref, regression_risks, imc_transition_counts, imc_transition_counts_ref, regression_ys, target_risks): 

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

    reg_auc = {}

    for reg_key in regression_risks.keys(): 
        fpr, tpr, threshold = metrics.roc_curve(alarms, regression_risks[reg_key])
        roc_auc = metrics.auc(fpr, tpr)
        reg_auc[reg_key] = roc_auc

    imc_results = {}

    for x in range(1,11):
        imc_results[str(x)] = []

    for key in imc_auc.keys():
        for entry in imc_results.keys():
            x = key.split('-')[0]
            if x == entry:
                imc_results[x].append(imc_auc[key])

    #print(imc_results)

    imc_ref_results = {}

    for x in range(1,11):
        imc_ref_results[str(x)] = []

    for key in imc_ref_auc.keys():
        for entry in imc_ref_results.keys():
            x = key.split('-')[0]
            if x == entry:
                imc_ref_results[x].append(imc_ref_auc[key])

    #print(imc_ref_results)

    reg_results = {}

    for x in range(1,11):
        reg_results[str(x)] = []

    for key in reg_auc.keys():
        for entry in reg_results.keys():
            x = key.split('-')[0]
            if x == entry:
                reg_results[x].append([key, reg_auc[key]])

    #print(reg_results)

    return target_auc, imc_results, imc_ref_results, reg_results, imc_transition_counts, imc_transition_counts_ref, regression_ys


def plotting(target_auc, imc_results, imc_ref_results, reg_results, imc_transition_counts, imc_transition_counts_ref, regression_ys): 

    R_sets = []
    NR_sets = []

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

    log = True

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

        # Plot mean line
        plt.plot(
            x_values,
            mean_auc,
            color='blue',
        )

        # Add shaded area for spread
        plt.fill_between(
            x_values,
            mean_auc - std_auc,
            mean_auc + std_auc,
            alpha=0.2,
            color='blue',
        )

    plt.plot(R_sets[0][0], R_sets[0][1], color='green')
    plt.plot(R_sets[1][0], R_sets[1][1], color='green')
    plt.plot(R_sets[2][0], R_sets[2][1], color='green')
    plt.plot(R_sets[3][0], R_sets[3][1], color='green')
    plt.plot(R_sets[4][0], R_sets[4][1], color='green')
    plt.plot(R_sets[5][0], R_sets[5][1], color='green')
    plt.plot(R_sets[6][0], R_sets[6][1], color='green')
    plt.plot(R_sets[7][0], R_sets[7][1], color='green')
    plt.plot(R_sets[8][0], R_sets[8][1], color='green')
    plt.plot(R_sets[9][0], R_sets[9][1], color='green')

    plt.xlabel("State count")
    plt.ylabel("AUC")
    if log:
        plt.yscale("log")
    else:
        plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.savefig("/workspaces/premise/premise/analysis/SnL_AUC_test.pdf", dpi=300)
    plt.show()


def main_imc(args: argparse.Namespace):

    suo, initial_amount, horizon = build_suo(args)

    length = initial_amount + horizon

    testing_samples = []
    for x in range(args.testing_samples):
            path = suo.generate_random_traces([], length)[0]
            testing_samples.append(tuple(path))

    alarms = aggregted_alarms(testing_samples)
    target_risks = stats_true(horizon, initial_amount, testing_samples, suo)

    imc_risks, imc_transition_counts = aggregated_stats_imc(args.imc_model, args.imc_stats, initial_amount, horizon, args, testing_samples)
    imc_risks_ref, imc_transition_counts_ref = aggregated_stats_imc(args.imc_model_ref, args.imc_stats_ref, initial_amount, horizon, args, testing_samples)
    regression_risks, regression_ys = aggregated_stats_regression(args.regression_model, args.regression_stats, testing_samples, horizon, initial_amount)
    target_auc, imc_results, imc_ref_results, reg_results, imc_transition_counts, imc_transition_counts_ref, regression_ys = auc_graph_prep(alarms, imc_risks, imc_risks_ref, regression_risks, imc_transition_counts, imc_transition_counts_ref, regression_ys, target_risks)
    plotting(target_auc, imc_results, imc_ref_results, reg_results, imc_transition_counts, imc_transition_counts_ref, regression_ys)


def build_learning_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Learning Parameters")

    group.add_argument("--model_name", type=str, default="MC", help="Name of the model (first letters code).")
    group.add_argument("-s", "--testing_samples", type=int, default = 25, help="Total number of samples used in learning")
    group.add_argument("--no-target", action="store_true", help="Do not use the target monitor" )


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

    parser.add_argument('--se_path', 
                        type = str, 
                        help = 'Path to state estimator',
    )

    parser.add_argument('--error_path', 
                        type = str, 
                        help ='Path to error estimator',
    )

    parser.add_argument('--rej_path', 
                        type = str, 
                        help ='Path to rejection classifier',
    )

    parser.add_argument('--stats_path',
                        type = str, 
                        help = 'Path to dataset statistics',
    )

    parser.add_argument('--cp_classification_path',
                        type = str, 
                        help = 'Path to CP classifier'
    )

    parser.add_argument('--imc_model',
                        type = str, 
                        help = 'Path imc models'
                        )
    
    parser.add_argument('--imc_stats',
                        type = str, 
                        help = 'Path imc stats'
    )
    parser.add_argument('--imc_model_ref',
                        type = str, 
                        help = 'Path imc models'
                        )
    parser.add_argument('--imc_stats_ref',
                        type = str, 
                        help = 'Path imc stats'
    )
    parser.add_argument('--regression_model',
                        type = str,
                        help = 'Path regression model'
    )
    parser.add_argument('--regression_stats', 
                        type = str,
                        help = 'Path regression stats'
    )

    return parser


if __name__ == "__main__":
    parser = testing_argsparser()
    args = parser.parse_args()
    main_imc(args)
    
    
#python -m premise.interval.batch_testing_AUC --mc SnL-10x10 --imc_model /workspaces/premise/out/models/2025-07-08_08-55-26/SnL-10x10-comp-no-ref --imc_stats /workspaces/premise/out/stats/2025-07-08_08-55-26/SnL-10x10-comp-noref-stats 

#python -m premise.interval.batch_testing_AUC --mc SnL-10x10 --imc_model /workspaces/premise/out/models/2025-07-08_08-55-26/SnL-10x10-comp-no-ref --imc_stats /workspaces/premise/out/stats/2025-07-08_08-55-26/SnL-10x10-comp-noref-stats --regression_model /workspaces/premise/out/models/2025-07-08_08-55-26/SnL-10x10-comp-reg --regression_stats /workspaces/premise/out/stats/2025-07-08_08-55-26/SnL-10x10-comp-reg-stats --imc_model_ref /workspaces/premise/out/models/2025-07-08_08-55-26/SnL-10x10-comp-ref --imc_stats_ref /workspaces/premise/out/stats/2025-07-08_08-55-26/SnL-10x10-comp-ref-stats


#python -m premise.interval.batch_testing_AUC --mc SnL-10x10 --imc_model /workspaces/premise/out/models/2025-07-08_19-02-12/SnL-10x10-comp-no-ref --imc_stats /workspaces/premise/out/stats/2025-07-08_19-02-12/SnL-10x10-comp-noref-stats --regression_model /workspaces/premise/out/models/2025-07-08_19-02-12/SnL-10x10-comp-reg --regression_stats /workspaces/premise/out/stats/2025-07-08_19-02-12/SnL-10x10-comp-reg-stats --imc_model_ref /workspaces/premise/out/models/2025-07-08_19-02-12/SnL-10x10-comp-ref --imc_stats_ref /workspaces/premise/out/stats/2025-07-08_19-02-12/SnL-10x10-comp-ref-stats







