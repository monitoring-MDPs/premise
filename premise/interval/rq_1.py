import argparse
from matplotlib import pyplot as plt
import numpy as np
from sympy import Line2D
from tqdm import tqdm, trange
import os
import matplotlib.ticker as ticker
import pickle
import numpy as np
from premise.models import *
from premise.interval.maximum_likelihood import *


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


def stats_true(horizon, initial_amount, testing_samples, suo):

    target_risks = []

    for trace in tqdm(testing_samples):
        sub_trace: Trace = trace[:initial_amount]

        target_risk, risks = test_monitor(
            suo.create_target_monitor(),
            [sub_trace],
            with_tqdm=False,
            intermediate_results=True,
        ) 

        target_risks.append(float(target_risk[sub_trace]))

    return target_risks


def aggregated_stats_imc(high_st, coarse, path, stats_path, initial_amount, horizon, args, testing_samples):
    imc_risks = {}

    imc_transition_counts = {}

    for x in range(1,11):
    #for x in range(6,7):
        print(f"Experiment number {x}")
        if coarse: 
            if high_st: 
                statistics = np.load(f"{stats_path}/high-st-{args.mc}-coarse_norefinement-stats-{x}.npy", allow_pickle=True)
            else:
                statistics = np.load(f"{stats_path}/{args.mc}-coarse_norefinement-stats-{x}.npy", allow_pickle=True)
        else:
            if high_st: 
                statistics = np.load(f"{stats_path}/high-st-{args.mc}-comp-noref-stats-{x}.npy", allow_pickle=True)
            else:
                print(os.path.getsize(f"{stats_path}/{args.mc}-comp-noref-stats-{x}.npy.npy"))
                statistics = np.load(f"{stats_path}/{args.mc}-comp-noref-stats-{x}.npy", allow_pickle=True)

        obj = statistics.item()
        imc_transition_count = obj["transitions_learned"]
        stopping_threashold = obj['args']['stopping_threshold']

        sum = 0
        total_imc_transition_count = []
        for z in range(len(imc_transition_count)):
            sum += imc_transition_count[z]
            total_imc_transition_count.append(sum)

        imc_transition_counts[str(x)] = total_imc_transition_count

        for y in trange(1, len(imc_transition_count) + 1):
            if coarse:
                if high_st:
                    initial_distribution = f"{path}/high-st-{args.mc}-coarse-comp-noref-{x}-{y}-initial_interval.npy"
                    transition_intervals = f"{path}/high-st-{args.mc}-coarse-comp-noref-{x}-{y}-interval.npy"
                else:
                    initial_distribution = f"{path}/{args.mc}-coarse-comp-noref-{x}-{y}-initial_interval.npy"
                    transition_intervals = f"{path}/{args.mc}-coarse-comp-noref-{x}-{y}-interval.npy"
            else: 
                if high_st: 
                    initial_distribution = f"{path}/high-st-{args.mc}-comp-noref-{x}-{y}-initial_interval.npy"
                    transition_intervals = f"{path}/high-st-{args.mc}-comp-noref-{x}-{y}-interval.npy"
                else: 
                    initial_distribution = f"{path}/{args.mc}-comp-noref-{x}-{y}-initial_interval.npy"
                    transition_intervals = f"{path}/{args.mc}-comp-noref-{x}-{y}-interval.npy"
                

            args.trans_path = transition_intervals
            args.init_path = initial_distribution
            args.dump = None
            args.exact = True 
            args.precision = 1e-6

            transition_intervals, initial_distribution = load_imc(args)

            mon, mon_comps = create_monitor(
                transition_intervals,
                initial_distribution,
                "min",
                True,
                horizon,
                args.dump
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

    return imc_risks, imc_transition_counts, stopping_threashold



def aggregated_stats_mc(high_st, coarse, path, stats_path, initial_amount, horizon, args, testing_samples):
    mc_risks = {}

    mc_transition_counts = {}

    for x in range(1, 11):
    #for x in range(6,7):
        print(f"Experiment number {x}")
        if coarse: 
            if high_st:
                statistics = np.load(f"{stats_path}/high-st-{args.mc}-coarse-comp-mle-stats-{x}.npy", allow_pickle=True)
            else:
                statistics = np.load(f"{stats_path}/{args.mc}-coarse-comp-mle-stats-{x}.npy", allow_pickle=True)
        else:
            if high_st: 
                statistics = np.load(f"{stats_path}/high-st-{args.mc}-comp-mle-stats-{x}.npy", allow_pickle=True)
            else:
                statistics = np.load(f"{stats_path}/{args.mc}-comp-mle-stats-{x}.npy", allow_pickle=True)

        mc_sample_count = statistics["sample_counts"]

        mc_transition_count = []

        for a in mc_sample_count: 
            mc_transition_count.append(a * (horizon + initial_amount))
    
        mc_transition_counts[str(x)] = mc_transition_count

        for y in trange(0, len(mc_transition_count)):
            if coarse:
                if high_st:
                    model = f"{path}/high-st-{args.mc}-coarse-comp-mle-{x}-{y}.pickl"
                else:
                    model = f"{path}/{args.mc}-coarse-comp-mle-{x}-{y}.pickl"
            else: 
                if high_st:
                    model = f"{path}/high-st-{args.mc}-comp-mle-{x}-{y}.pickl"
                else:
                    model = f"{path}/{args.mc}-comp-mle-{x}-{y}.pickl"

            with open(model, 'rb') as file:
                data = pickle.load(file)

            model, observation_map, state_index_map = dict_to_pomdp(data[1], data[0], target_label =True, use_exact=True) 

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


def distance_graph(high_st, coarse,target_risks, imc_risks, imc_transition_counts, mc_risks, mc_transition_counts, testing_samples_weights, stopping_threashold):

    #IMC 
    distance_stats = {}

    for key in imc_risks.keys():
        total_distance = 0
        for x in range(len(target_risks)):
            #total_distance += testing_samples_weights[x] * abs(
            #    imc_risks[key][x] - target_risks[x]
            #)
            total_distance += abs(
               imc_risks[key][x] - target_risks[x]
            )
        distance_stats[key] = total_distance / args.testing_samples

    distance_graph_data = {}
    for x in range(1, 11):
    #for x in range(6,7):
        distance_graph_data[x] = []

    for key in distance_stats.keys():
        for x in range(1, 11):
        #for x in range(6,7):
            if key.split("-")[0] == str(x):
                distance_graph_data[x].append(distance_stats[key])


    final_distances = []
    for x in range(1,11):
    #for x in range(6,7):
        final_distances.append(distance_graph_data[x][-1]) 


    graph_data = []
    for x in range(1,11):
    #for x in range(6,7):
        graph_data.append([distance_graph_data[x], imc_transition_counts[str(x)]])

  
    #MC

    mc_distance_stats = {}

    for key in mc_risks.keys():
        total_distance = 0
        for x in range(len(target_risks)):
            #total_distance += testing_samples_weights[x] * abs(
            #    mc_risks[key][x] - target_risks[x]
            #)
            total_distance += abs(
               mc_risks[key][x] - target_risks[x]
            )
        mc_distance_stats[key] = total_distance / args.testing_samples


    mc_distance_graph_data = {}
    for x in range(1, 11):
    #for x in range(6,7):
        mc_distance_graph_data[x] = []

    for key in mc_distance_stats.keys():
        for x in range(1, 11):
        #for x in range(6,7):
            if key.split("-")[0] == str(x):
                mc_distance_graph_data[x].append(mc_distance_stats[key])
    

    mc_final_distances = []
    for x in range(1,11):
    #for x in range(6,7):
        mc_final_distances.append(mc_distance_graph_data[x][-1]) 


    mc_graph_data = []
    for x in range(1,11):
    #for x in range(6,7):
        mc_graph_data.append([mc_distance_graph_data[x], mc_transition_counts[str(x)]])

    log = False
    plt.figure()
    fig, ax = plt.subplots(figsize=(16, 8))

    #MC AVERAGE PERFORMANCE

    mc_transitions_data = []
    mc_distance_data = []

    for entry in mc_graph_data:
        transitions = entry[1]
        distance_daum = entry[0]

        mc_transitions_data.append(transitions)
        mc_distance_data.append(distance_daum)

    # Find common x range for interpolation
    min_x = max(min(transitions) for transitions in mc_transitions_data)
    max_x = min(max(transitions) for transitions in mc_transitions_data)
    mc_x_values = np.linspace(min_x, max_x, 500)

    # Interpolate all runs to common x values
    mc_interpolated_auc = []
    for auc, transitions in zip(mc_distance_data, mc_transitions_data):
        if log:
            auc = np.log10(auc)
        interpolated = np.interp(mc_x_values, transitions, auc)
        if log:
            interpolated = np.power(10, interpolated)
        mc_interpolated_auc.append(interpolated)

    # Calculate mean and std for interpolated y values
    mc_distance_array = np.array(mc_interpolated_auc)
    mc_mean_distance = np.mean(mc_distance_array, axis=0)
    mc_std_distance = np.std(mc_distance_array, axis=0)

    # Plot mean line
    ax.plot(
        mc_x_values,
        mc_mean_distance,
        color='chartreuse',
        label = f'MC, (Average final distance: {np.mean(mc_final_distances):.3f})',
        linewidth=7,
        linestyle=':',
        )

    # Add shaded area for spread
    ax.fill_between(
            mc_x_values,
            mc_mean_distance - mc_std_distance,
            mc_mean_distance + mc_std_distance,
            alpha=0.2,
            color='chartreuse',
        )

    # IMC AVERAGE PERFORMANCE
    transitions_data = []
    distance_data = []

    for entry in graph_data:
        transitions = entry[1]
        distance_daum = entry[0]

        transitions_data.append(transitions)
        distance_data.append(distance_daum)

    # Find common x range for interpolation
    min_x = max(min(transitions) for transitions in transitions_data)
    max_x = min(max(transitions) for transitions in transitions_data)
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
            color='red',
            label = f'IMC, (Average final distance: {np.mean(final_distances):.3f})',
            linewidth=7,
            linestyle='--',
            )

    # Add shaded area for spread
    ax.fill_between(
                x_values,
                mean_distance - std_distance,
                mean_distance + std_distance,
                alpha=0.2,
                color='red',
            ) 
    
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((4, 4))  # Force 10^4 scale
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='both', labelsize=25)
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
    if coarse: 
        if high_st: 
            plt.title(f'{args.mc} coarse, stopping threashold: {stopping_threashold}', fontsize=30)
        else:
            plt.title(f'{args.mc} coarse', fontsize=30)
    else: 
        if high_st:
            plt.title(f'{args.mc},  stopping threashold: {stopping_threashold}', fontsize=30)
        else:
            plt.title(f'{args.mc}', fontsize=30)

    plt.tight_layout()
    if coarse: 
        if high_st:
            plt.savefig(f"/workspaces/premise/premise/analysis/rq_1_high-st-{args.mc}_coarse_distance_to_RRF.pdf", dpi=300,  bbox_inches='tight')
        else: 
            plt.savefig(f"/workspaces/premise/premise/analysis/rq_1_{args.mc}_coarse_distance_to_RRF.pdf", dpi=300,  bbox_inches='tight')
    else: 
        if high_st:
            plt.savefig(f"/workspaces/premise/premise/analysis/rq_1_high-st-{args.mc}_distance_to_RRF.pdf", dpi=300,  bbox_inches='tight')
        else:
            plt.savefig(f"/workspaces/premise/premise/analysis/rq_1_{args.mc}_distance_to_RRF.pdf", dpi=300,  bbox_inches='tight')

    plt.show()


def overestimation_graph(high_st, coarse, target_risks, imc_risks, imc_transition_counts, mc_risks, mc_transition_counts, stopping_threashold):

    plt.figure()
    plt.plot([0, 1], [0, 1], "--", color = 'black')

    mc_ys = []

    for key in mc_risks.keys():
        mc_ys.append(int(key.split('-')[1]))

    for key in mc_risks.keys():
        if key.split('-')[1] == str(max(mc_ys)):
            if key.split('-')[0] == str(5):
                plt.scatter(mc_risks[key], target_risks, color='chartreuse', marker='s', label = "MC")
            else: 
                plt.scatter(mc_risks[key], target_risks, color='chartreuse', marker='s')

    
    imc_ys = []

    for key in imc_risks.keys():
        imc_ys.append(int(key.split('-')[1]))


    for key in imc_risks.keys():
        if key.split('-')[1] == str(max(imc_ys)):
            if key.split('-')[0] == str(5):
                plt.scatter(imc_risks[key], target_risks, color='red', marker='o', label = "IMC")
            else: 
                plt.scatter(imc_risks[key], target_risks, color='red', marker='o')


    plt.legend(fontsize=15)
  
    plt.xlabel("IMC and MC risks", fontsize=15)
    plt.ylabel("Target risks", fontsize=15)
    plt.tick_params(axis='both', labelsize=12)
    if coarse:
        if high_st:
            plt.title(f'{args.mc} coarse, stopping threashold: {stopping_threashold}', fontsize=15)
        else:
            plt.title(f'{args.mc} coarse', fontsize=15)
    else:
        if high_st:
            plt.title(f'{args.mc}, stopping threashold: {stopping_threashold}', fontsize=15)
        else: 
            plt.title(f'{args.mc}', fontsize=15)

    plt.tight_layout()
    if coarse:
        if high_st:
            plt.savefig(f"/workspaces/premise/premise/analysis/rq_1_high-st{args.mc}_coarse_overestimation.pdf", dpi=300)
        else:
            plt.savefig(f"/workspaces/premise/premise/analysis/rq_1_{args.mc}_coarse_overestimation.pdf", dpi=300)
    else: 
        if high_st:
            plt.savefig(f"/workspaces/premise/premise/analysis/rq_1_high-st{args.mc}_overestimation.pdf", dpi=300)
        else:
            plt.savefig(f"/workspaces/premise/premise/analysis/rq_1_{args.mc}_overestimation.pdf", dpi=300)

    plt.show()


def main_imc(args: argparse.Namespace):

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
    else:
        coarse = False

    


    target_risks = stats_true(horizon, initial_amount, testing_samples, suo)
    imc_risks, imc_transition_counts, stopping_threashold = aggregated_stats_imc(args.high_st, coarse, args.model_path, args.stats_path, initial_amount, horizon, args, testing_samples)
    mc_risks, mc_transition_counts = aggregated_stats_mc(args.high_st, coarse, args.model_path, args.stats_path, initial_amount, horizon, args, testing_samples)

    distance_graph(args.high_st, coarse, target_risks, imc_risks, imc_transition_counts, mc_risks, mc_transition_counts, testing_samples_weights, stopping_threashold)
    overestimation_graph(args.high_st, coarse, target_risks, imc_risks, imc_transition_counts, mc_risks, mc_transition_counts, stopping_threashold)


def build_learning_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Learning Parameters")

    group.add_argument("--model_name", type=str, default="MC", help="Name of the model (first letters code).")
    group.add_argument("-s", "--testing_samples", type=int, default = 50, help="Total number of samples used in learning")


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

    parser.add_argument("--model_path", type=str, help="Path models")
    parser.add_argument("--stats_path", type=str, help="Path stats")
    parser.add_argument("--high_st", type = bool, default = False, help = "If higher stopping threashold is used")

    return parser


if __name__ == "__main__":
    parser = testing_argsparser()
    args = parser.parse_args()
    main_imc(args)



#python -m premise.interval.rq_1 --mc airportA-7-10-10 -sv d p pobs turn --model_path /workspaces/premise/out/models/2025-07-17 --stats_path /workspaces/premise/out/stats/2025-07-17 
#python -m premise.interval.rq_1 --mc airportA-7-10-10 --model_path /workspaces/premise/out/models/2025-07-19 --stats_path /workspaces/premise/out/stats/2025-07-19 

#python -m premise.interval.rq_1 --mc SnL-10x10 --model_path /workspaces/premise/out/models/2025-07-17 --stats_path /workspaces/premise/out/stats/2025-07-17


#python -m premise.interval.rq_1 --mc SnL-10x10 --model_path /workspaces/premise/out/models/2025-07-23_21-19-00 --stats_path /workspaces/premise/out/stats/2025-07-23_21-19-00