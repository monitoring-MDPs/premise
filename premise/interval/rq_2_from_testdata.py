# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
from premise.interval.loading import build_suo_args_parser
import os
import pickle

from premise.interval.utils import setup_logging


def probelm_statement(high_st, coarse, imc_data, imc_final_distances, ref_data, ref_final_distances, split_ref_data, split_ref_final_distances, out_path):
 
    log = False

    plt.figure()
    fig, ax = plt.subplots(figsize=(16, 8))

    # NO REFINEMENT AVERAGE PERFORMANCE
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

    # Plot mean line
    ax.plot(
        x_values,
        distance_mean,
        color="red",
        label=f"No refinement, (Average final distance: {np.mean(imc_final_distances):.4f})",
        linewidth=3,
        linestyle= ':', 
        marker ='P', 
        markersize=6,
        markevery=20
    )

    # Add shaded area for spread
    ax.fill_between(
        x_values,
        distance_mean - distance_std,
        distance_mean + distance_std,
        alpha=0.2,
        color="red",
    )

    # REFINEMENT AVERAGE PERFORMANCE
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
    for auc, transitions in zip(R_distance_data, R_transitions_data):
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

    # Plot mean line
    ax.plot(
        x_values,
        R_distance_mean,
        color="blue",
        label=f"Refinement, (Average final width: {np.mean(ref_final_distances):.4f})",
        linewidth=3,
        linestyle= ':', 
        marker ='o', 
        markersize=6,
        markevery=20
    )

    # Add shaded area for spread
    ax.fill_between(
        x_values,
        R_distance_mean - R_distance_std,
        R_distance_mean + R_distance_std,
        alpha=0.2,
        color="blue",
    )

    # SPLITTING REFINEMENT AVERAGE PERFORMANCE
    SR_transitions_data = []
    SR_distance_data = []

    for key in split_ref_data.keys():
        sref_transitions = split_ref_data[key][1]
        sref_distance = split_ref_data[key][0]

        SR_transitions_data.append(sref_transitions)
        SR_distance_data.append(sref_distance)

    # Find common x range for interpolation
    SR_min_x = max(min(transitions) for transitions in SR_transitions_data)
    SR_max_x = min(max(transitions) for transitions in SR_transitions_data)
    SR_x_values = np.linspace(SR_min_x, SR_max_x, 500)

    # Interpolate all runs to common x values
    SR_interpolated_data = []
    for auc, transitions in zip(SR_distance_data, SR_transitions_data):
        if log:
            auc = np.log10(auc)
        SR_interpolated = np.interp(SR_x_values, transitions, auc)
        if log:
            SR_interpolated = np.power(10, SR_interpolated)
        SR_interpolated_data.append(SR_interpolated)

    # Calculate mean and std for interpolated y values
    SR_distance_array = np.array(SR_interpolated_data)
    SR_distance_mean = np.mean(SR_distance_array, axis=0)
    SR_distance_std = np.std(SR_distance_array, axis=0)

    # Plot mean line
    ax.plot(
        x_values,
        SR_distance_mean,
        color="aqua",
        label=f"Refinement with splitting, (Average final width: {np.mean(ref_final_distances):.4f})",
        linewidth=3,
        linestyle= ':',
        marker ='s', 
        markersize=6, 
        markevery=20
    )

    # Add shaded area for spread
    ax.fill_between(
        x_values,
        SR_distance_mean - SR_distance_std,
        SR_distance_mean + SR_distance_std,
        alpha=0.2,
        color="aqua",
    )


    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((4, 4))  # Force 10^4 scale
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="both", labelsize=25)
    ax.xaxis.get_offset_text().set_size(25)

    ax.set_xlabel("Explored states", fontsize=25)
    ax.set_ylabel("Interval Width", fontsize=25)
    ax.legend(loc="upper right", fontsize=25)
    if log:
        plt.yscale("log")
    else:
        plt.ylim(bottom=0)
    ax.grid(True)
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout() 

    if coarse:
        if high_st:
            plt.savefig(
                f"{out_path}/r2_high-st-{args.mc}_coarse_interval_width.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                f"{out_path}/r2_{args.mc}_coarse_interval_width.pdf",
                dpi=300,
                bbox_inches="tight",
            )
    else:
        if high_st:
            plt.savefig(
                f"{out_path}/r2_high-st-{args.mc}_interval_width.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                f"{out_path}/r2_{args.mc}_interval_width.pdf",
                dpi=300,
                bbox_inches="tight",
            )

    plt.show()


def main(args: argparse.Namespace):
    setup_logging("rq2:" + args.mc)

    os.makedirs(args.out, exist_ok=True)

    with open(args.testdata_rq_2, 'rb') as f:
        data = pickle.load(f)


    model = data['model']
    coarse = data['coarse'] 
    high_st = data['high_st'] 
    imc_data = data['imc_data'] 
    imc_final_distances = data['imc_final_distances'] 
    ref_data = data['ref_data'] 
    ref_final_distances = data['ref_final_distances'] 
    split_ref_data  = data['split_ref_data'] 
    split_ref_final_distances = data['split_ref_final_distances'] 

    
    probelm_statement(high_st, coarse, imc_data, imc_final_distances, ref_data, ref_final_distances, split_ref_data, split_ref_final_distances, args.out)


def testing_argsparser():
    parser = argparse.ArgumentParser(description="Learn an IMC")
    build_suo_args_parser(parser)

    parser.add_argument(
        "--testdata_rq_2", type=str, help="Path to stats")
    
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
    main(args)
