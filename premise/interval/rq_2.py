# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
from premise.interval.loading import build_suo_args_parser

def probelm_statement(imc_path, ref_path):

    imc_data = {}
    imc_final_distances = []

    for x in range(1,11): 
        states_aggreagted = []
        print(f'Experiment number {x}')
        statistics = np.load(f'{imc_path}-{x}.npy', allow_pickle=True).item()
    
        distances = statistics["distances"]
        imc_final_distances.append(distances[-1])
        states_vistited = statistics["transitions_learned"]
        total = 0 
        for s in states_vistited: 
            total += s
            states_aggreagted.append(total)

        print(states_aggreagted)
        
        imc_data[x] = [distances, states_aggreagted]

    ref_data = {}
    ref_final_distances = []
    
    for x in range(1,11): 
        ref_states_aggreagted = []
        print(f'Experiment number {x}')
        statistics = np.load(f'{ref_path}-{x}.npy', allow_pickle=True).item()

        distances = statistics["distances"]
        ref_final_distances.append(distances[-1])
        states_vistited = statistics["transitions_learned"]
        total = 0 
        for s in states_vistited: 
            total += s
            ref_states_aggreagted.append(total)
        
        print(ref_states_aggreagted)

        ref_data[x] = [distances, ref_states_aggreagted]

    
    log = True

    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 5))

    #NO REFINEMENT AVERAGE PERFORMANCE
    transitions_data = []
    distance_data = []

    for key in imc_data.keys():
        transitions = imc_data[key][1]
        distance = imc_data[key][0]

        transitions_data.append(transitions)
        distance_data.append(distance)

    # Find common x range for interpolation
    min_x = max(min(transitions) for transitions in transitions_data)
    max_x = min(max(transitions) for transitions in transitions_data)
    x_values = np.linspace(min_x, max_x, 500)

    # Interpolate all runs to common x values
    interpolated_data = []
    for auc, transitions in zip(distance_data, transitions_data):
        if log:
            auc = np.log10(auc)
        interpolated = np.interp(x_values, transitions, auc)
        if log:
            interpolated = np.power(10, interpolated)
        interpolated_data.append(interpolated)

    # Calculate mean and std for interpolated y values
    distance_array = np.array(interpolated_data)
    distance_mean = np.mean(distance_array, axis=0)
    distance_std = np.std(distance_array, axis=0)

    #Plot mean line
    ax.plot(
        x_values,
        distance_mean,
        color='red',
        label = f'No refinement, (Mean final distance: {np.mean(imc_final_distances):.3f})',
        linewidth=3,
        linestyle='--',
        )

    #Add shaded area for spread
    ax.fill_between(
            x_values,
            distance_mean - distance_std,
            distance_mean + distance_std,
            alpha=0.2,
            color='red',
        )

    #REFINEMENT AVERAGE PERFORMANCE
    R_transitions_data = []
    R_distance_data = []

    for key in ref_data.keys():
        ref_transitions = ref_data[key][1]
        ref_distance = ref_data[key][0]

        R_transitions_data.append(ref_transitions)
        R_distance_data.append(ref_distance)

    # Find common x range for interpolation
    R_min_x = max(min(transitions) for transitions in R_transitions_data)
    R_max_x = min(max(transitions) for transitions in R_transitions_data)
    R_x_values = np.linspace(R_min_x, R_max_x, 500)

    # Interpolate all runs to common x values
    R_interpolated_data = []
    for auc, transitions in zip(distance_data, transitions_data):
        if log:
            auc = np.log10(auc)
        R_interpolated = np.interp(R_x_values, transitions, auc)
        if log:
            R_interpolated = np.power(10, R_interpolated)
        R_interpolated_data.append(R_interpolated)

    # Calculate mean and std for interpolated y values
    R_distance_array = np.array(R_interpolated_data)
    R_distance_mean = np.mean(R_distance_array, axis=0)
    R_distance_std = np.std(R_distance_array, axis=0)

    #Plot mean line
    ax.plot(
        x_values,
        R_distance_mean,
        color='blue',
        label = f'Refinement, (Mean final distance: {np.mean(ref_final_distances):.3f})',
        linewidth=3,
        linestyle=':',
        )

    #Add shaded area for spread
    ax.fill_between(
            x_values,
            R_distance_mean - R_distance_std,
            R_distance_mean + R_distance_std,
            alpha=0.2,
            color='blue',
        )
    
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((4, 4))  # Force 10^4 scale
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='both', labelsize=15)
    ax.xaxis.get_offset_text().set_size(15)

    ax.set_xlabel("State count", fontsize=20)
    ax.set_ylabel("Interval Width", fontsize=20)
    ax.legend(loc="upper right", fontsize=15)
    if log:
        plt.yscale("log")
    else:
        plt.ylim(bottom=0)
    ax.grid(True)
    plt.subplots_adjust(bottom=0.25)
    plt.title(f'{args.mc}', fontsize=20)

    plt.savefig("/workspaces/premise/premise/analysis/Distance_test.pdf", dpi=300)
    plt.show()


def main(args: argparse.Namespace):
    imc_stats = args.imc_stats
    imc_stats_ref = args.imc_stats_ref

    probelm_statement(imc_stats, imc_stats_ref)

def testing_argsparser():
    parser = argparse.ArgumentParser(description="Learn an IMC")
    build_suo_args_parser(parser)


    parser.add_argument('--imc_stats',
                        type = str, 
                        help = 'Path imc stats'
    )
   
    parser.add_argument('--imc_stats_ref',
                        type = str, 
                        help = 'Path imc stats'
    )

    return parser


if __name__ == "__main__":
    parser = testing_argsparser()
    args = parser.parse_args()
    main(args)


#python -m premise.interval.ref_vs_no_ref --mc SnL-10x10 --imc_stats /workspaces/premise/out/stats/2025-07-08_19-02-12/SnL-10x10-comp-noref-stats --imc_stats_ref /workspaces/premise/out/stats/2025-07-08_19-02-12/SnL-10x10-comp-ref-stats

