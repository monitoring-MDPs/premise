#!/usr/bin/env python3
"""
Plot benchmark results from JSON file.
"""

import argparse
from email.mime import base
from itertools import product
import json
from math import comb
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from benchmark import PROPERTIES

# Configuration: Colormap for model colors
COLORMAP_NAME = "tab20"  # Can be changed to any matplotlib discrete colormap (e.g., 'tab10', 'Set3', 'Paired')

# Arithmetic mode display names
ARITHMETIC_MODE_NAMES = {
    "exact": "exact",
    "float": "float",
    "exact-tolerance": "$\\varepsilon$-exact",
}

# Query type display names
QUERY_TYPE_NAMES = {
    "quantitative": "quantitative",
    "bounded": "qualitative",
}

# Model display names (maps internal names to display names)
# Only list models that need renaming; others will use their internal name
MODEL_NAMES = {
    # Add custom model name mappings here
    "brp-N=16-MAX=8-PCHAN=0.010": "BRP-16-0.010",
    "brp-N=16-MAX=8-PCHAN=0.015": "BRP-16-0.015",
    "brp-N=16-MAX=8-PCHAN=0.020": "BRP-16-0.020",
    "brp-N=16-MAX=8-PCHAN=0.025": "BRP-16-0.025",
    "brp-N=16-MAX=8-PCHAN=0.030": "BRP-16-0.030",
    "brp-N=16-MAX=8-PCHAN=0.035": "BRP-16-0.035",
    "brp-N=16-MAX=8-PCHAN=0.040": "BRP-16-0.040",
    "brp-N=16-MAX=8-PCHAN=0.045": "BRP-16-0.045",
    "brp-N=16-MAX=8-PCHAN=0.050": "BRP-16-0.050",
    "brp-N=32-MAX=9-PCHAN=0.010": "BRP-32-0.010",
    "brp-N=32-MAX=9-PCHAN=0.015": "BRP-32-0.015",
    "brp-N=32-MAX=9-PCHAN=0.020": "BRP-32-0.020",
    "brp-N=32-MAX=9-PCHAN=0.025": "BRP-32-0.025",
    "brp-N=32-MAX=9-PCHAN=0.030": "BRP-32-0.030",
    "brp-N=32-MAX=9-PCHAN=0.035": "BRP-32-0.035",
    "brp-N=32-MAX=9-PCHAN=0.040": "BRP-32-0.040",
    "brp-N=32-MAX=9-PCHAN=0.045": "BRP-32-0.045",
    "brp-N=32-MAX=9-PCHAN=0.050": "BRP-32-0.050",
    "brp-N=64-MAX=10-PCHAN=0.010": "BRP-64-0.010",
    "brp-N=64-MAX=10-PCHAN=0.015": "BRP-64-0.015",
    "brp-N=64-MAX=10-PCHAN=0.020": "BRP-64-0.020",
    "brp-N=64-MAX=10-PCHAN=0.025": "BRP-64-0.025",
    "brp-N=64-MAX=10-PCHAN=0.030": "BRP-64-0.030",
    "brp-N=64-MAX=10-PCHAN=0.035": "BRP-64-0.035",
    "brp-N=64-MAX=10-PCHAN=0.040": "BRP-64-0.040",
    "brp-N=64-MAX=10-PCHAN=0.045": "BRP-64-0.045",
    "brp-N=64-MAX=10-PCHAN=0.050": "BRP-64-0.050",
}

bad_names = []


def get_arithmetic_mode_name(mode):
    """Get the display name for an arithmetic mode.

    Args:
        mode: Internal arithmetic mode name

    Returns:
        Display name for the mode
    """
    return ARITHMETIC_MODE_NAMES[mode]


def get_query_type_name(query_type):
    """Get the display name for a query type.

    Args:
        query_type: Internal query type name

    Returns:
        Display name for the query type
    """
    return QUERY_TYPE_NAMES[query_type]


def get_model_name(model_name):
    """Get the display name for a model.

    Args:
        model_name: Internal model name

    Returns:
        Display name for the model (with underscores escaped for LaTeX)
    """
    # Get custom name if defined, otherwise use original
    display_name = MODEL_NAMES.get(model_name, model_name)
    # Escape underscores for LaTeX
    return display_name.replace("_", "\\_")


def get_model_colors(models):
    """Get colors for models using the configured colormap.

    Args:
        models: List of model names

    Returns:
        Dictionary mapping model names to colors
    """
    import matplotlib

    cmap = matplotlib.colormaps.get_cmap(COLORMAP_NAME)
    num_models = len(models)
    # For qualitative colormaps like tab20, use discrete indices
    colors = [cmap(i % cmap.N) for i in range(num_models)]
    return {model: colors[i] for i, model in enumerate(models)}


def get_properties_from_results(results):
    """Extract properties from results data, grouped by model."""
    model_props = {}
    for r in results:
        model = r["model"]
        path_formula = r["path_formula"]
        if model and path_formula:
            if model not in model_props:
                model_props[model] = set()
            model_props[model].add(path_formula)
    # Convert sets to sorted lists
    return {model: sorted(props) for model, props in model_props.items()}


def exclude_model(model_name: str, plot_type: str):
    """Determine if a model should be excluded from a specific plot type.

    Args:
        model_name: Name of the model
        plot_type: Type of plot ('scatter', 'heatmap', 'speedup_vs_marginal')

    Returns:
        True if model should be excluded, False otherwise
    """
    if model_name.startswith("brp-N=128-MAX=11"):
        return True
    # allowed_brp = [
    #     "brp-N=16-MAX=8-PCHAN=0.010",
    #     "brp-N=32-MAX=9-PCHAN=0.010",
    #     "brp-N=64-MAX=10-PCHAN=0.010",
    # ]
    #
    # if plot_type != "marginal":
    #     if model_name.startswith("brp-") and "0.010" not in model_name:
    #         return True

    return False


def load_results(json_file):
    """Load benchmark results from JSON file."""
    with open(json_file, "r") as f:
        return list(r for r in json.load(f) if r is not None and not r["unfinished"])


def validate_results(results):
    """Validate results by majority voting on exact values and checking float consistency.

    Returns a dict mapping (model, method, arithmetic_mode, query_type) -> bool indicating correctness.
    """
    from collections import Counter

    correctness = {}
    wrong_by = {}



    # Group results by model and query_type
    by_model_query = {}
    for r in results:
        if not r["success"] or r["timeout"]:
            continue

        skip_bad = False
        for bad_name in bad_names:
            print(r["model"])
            if r["model"].startswith(bad_name):
                print("Skipping bad name '{}'".format(bad_name))
                skip_bad = True
        if skip_bad:
            continue


        key = (r["model"], r["query_type"], r["path_formula"])
        if key not in by_model_query:
            by_model_query[key] = []
        by_model_query[key].append(r)

    for (model, query_type, path_formula), group_results in by_model_query.items():
        if query_type == "quantitative":
            # For quantitative queries, use majority voting on exact arithmetic values
            exact_results = [
                r for r in group_results if r["arithmetic_mode"] == "exact"
            ]

            if exact_results:
                # Get majority value (round to 10 decimal places for comparison)
                exact_values = [round(r["value"], 10) for r in exact_results]
                value_counts = Counter(exact_values)
                majority_value, count = value_counts.most_common(1)[0]
                if count < 2 and len(value_counts) > 1:
                    print(
                        f"No majority value for model {model}, query {query_type}, path_formula {path_formula}: {value_counts}"
                    )
            #else:
                # No force-exact results, fall back to exact
                # exact_results = [
                #     r for r in group_results if r["arithmetic_mode"] == "exact-tolerance"
                # ]
                # majority_value = np.average(
                #     [round(r["value"], 10) for r in exact_results]
                # )
                # print(
                #     f"WARNING, no exact results for model {model}, query {query_type}, path_formula {path_formula}, using exact-tolerance average {majority_value} for validation."
                # )

            # Mark all results as correct/incorrect based on majority
            for r in group_results:
                result_key = (
                    r["model"],
                    r["method"],
                    r["arithmetic_mode"],
                    r["query_type"],
                    r["path_formula"],
                )
                rounded_value = round(r["value"], 10)

                if r["arithmetic_mode"] == "exact":
                    correctness[result_key] = rounded_value == majority_value
                    if not correctness[result_key]:
                        # This is very suspect, as exact arithmetic should agree
                        print(
                            f"Discrepancy in exact arithmetic for {result_key}: value {rounded_value} vs majority {majority_value}. All exact results: { {r['method']: r['value'] for r in exact_results} }"
                        )
                else:  # float or imprecise exact
                    # Float values should be close to majority (within 1e-5)
                    if r["value"] < 1e-8 and majority_value < 1e-8:
                        correctness[result_key] = True
                    else:
                        correct = abs(r["value"] - majority_value) / majority_value < 1e-3
                        correctness[result_key] = correct


                if not correctness[result_key]:
                    print(f"{r["value"]} vs {majority_value}")
                    minimum = min(r["value"], majority_value)
                    if minimum == 0:
                        wrong_by[result_key] = "inf"
                    else:
                        wrong_by[result_key] = abs(r["value"] - majority_value) / min(r["value"], majority_value)

        elif query_type == "bounded":
            # For bounded queries, check if they match the threshold comparison
            # First get the correct quantitative value for this model
            majority_quant_value =  None
            if (model, "quantitative", path_formula) in by_model_query:
                quant_results = by_model_query[(model, "quantitative", path_formula)]
                exact_quant = [
                    r for r in quant_results if r["arithmetic_mode"] == "exact"
                ]
                # Get majority quantitative value
                exact_values = [round(r["value"], 10) for r in exact_quant]
                value_counts = Counter(exact_values)
                if len(value_counts) > 0:
                    majority_quant_value, _ = value_counts.most_common(1)[0]


            if majority_quant_value is None:
                print(
                    f"WARNING, no exact quantitative results for model {model}."
                )

            # For each bounded result, check if it matches the expected boolean
            for r in group_results:
                result_key = (
                    r["model"],
                    r["method"],
                    r["arithmetic_mode"],
                    r["query_type"],
                    r["path_formula"],
                )
                threshold = r["threshold"]
                if majority_quant_value is not None:
                    expected_value = 1.0 if majority_quant_value >= threshold else 0.0
                    correctness[result_key] = r["value"] == expected_value
                    if not correctness[result_key]:
                        print(f"result: {r["value"]} and truth: {majority_quant_value} >= {threshold}")

                        wrong_by[result_key] = abs(r["value"] - expected_value)
                else:
                    correctness[result_key] = None



    return correctness, wrong_by


def get_model_source(model_name):
    """Determine which folder/source a model came from.

    Args:
        model_name: Name of the model

    Returns:
        Tuple of (source_name, sort_order)
    """
    # BN benchmarks: Bayesian network benchmarks
    bn_models = [
        "alarm",
        "andes",
        "asia",
        "barley",
        "cancer",
        "child",
        "earthquake",
        "hailfinder",
        "hepar2",
        "insurance",
        "pathfinder",
        "sachs",
        "survey",
        "win95pts",
        "water",
    ]
    for bn in bn_models:
        if model_name.startswith(bn):
            return ("BN-benchmarks", 1)

    # Transformed MDPs: brp, crowds variations
    if model_name.startswith("brp") or model_name.startswith("crowds"):
        return ("Transformed-MDP", 2)

    # Concrete MDPs: wlan, other concrete models
    if model_name.startswith("wlan") or model_name.startswith("coin"):
        return ("Concrete-MDP", 3)

    # Monitoring MDPs
    return ("Monitoring-MDP", 4)


def generate_latex_table(results, output_file, correctness, query_type="quantitative"):
    """Generate a LaTeX table comparing runtimes for all models, methods, and arithmetic modes.

    Args:
        results: List of benchmark results
        output_file: Path to output .tex file
        correctness: Correctness dict from validate_results
        query_type: 'quantitative' or 'bounded'
    """
    # Filter results by query type
    filtered_results = [r for r in results if r["query_type"] == query_type]

    if not filtered_results:
        print(f"No results for query type {query_type}")
        return

    # Get all unique models, methods, and arithmetic modes
    all_models = sorted(
        set(
            r["model"]
            for r in filtered_results
            if not exclude_model(r["model"], "table")
        )
    )
    methods = sorted(set(r["method"] for r in filtered_results))
    arithmetic_modes = sorted(set(r["arithmetic_mode"] for r in filtered_results))

    # Sort models by source, then by name
    models_with_source = [(m, get_model_source(m)) for m in all_models]
    models_with_source.sort(
        key=lambda x: (x[1][1], x[0])
    )  # Sort by source order, then name
    models = [m for m, _ in models_with_source]
    model_sources = {m: src for m, (src, _) in models_with_source}

    # Get properties per model
    properties_map = get_properties_from_results(filtered_results)

    # Create method×arithmetic combinations
    configs = [(method, arith) for method in methods for arith in arithmetic_modes]

    # Build data structure: model -> property -> config -> (time, is_correct, is_timeout)
    # Also collect model stats (states, transitions) and property marginals
    model_stats = {}  # model -> (states, transitions)
    property_marginals = {}  # (model, prop) -> marginal

    data = {}
    for model in models:
        data[model] = {}
        if model not in properties_map:
            raise ValueError(f"Model {model} not found in properties_map")
        for prop in properties_map[model]:
            data[model][prop] = {}
            for method, arith in configs:
                matching = [
                    r
                    for r in filtered_results
                    if r["model"] == model
                    and r["path_formula"] == prop
                    and r["method"] == method
                    and r["arithmetic_mode"] == arith
                ]
                if matching:
                    r = matching[0]
                    key = (model, method, arith, query_type, prop)
                    is_timeout = r["timeout"]
                    if is_timeout:
                        is_correct = None
                    else:
                        if key in correctness:
                            is_correct = correctness[key]
                        else:
                            print(f"WARNING: key {key} not found in correctness.")
                            is_correct = None

                    quan_key = (model, method, arith, "quantitative", prop)
                    if quan_key in correctness:
                        is_quan_correct = correctness[
                            (model, method, arith, "quantitative", prop)
                        ]
                    else:
                        is_quan_correct = None

                    time_val = r["time"]
                    data[model][prop][(method, arith)] = (
                        time_val,
                        is_correct,
                        is_quan_correct,
                        is_timeout,
                    )

                    # Store model stats (same for all properties)
                    if (
                        model not in model_stats
                        and "states" in r
                        and "transitions" in r
                    ):
                        model_stats[model] = (r["states"], r["transitions"])

                    # Store property marginal
                    if (model, prop) not in property_marginals and "marginal" in r:
                        property_marginals[(model, prop)] = r["marginal"]

    # Start building LaTeX table
    lines = []

    # Calculate number of columns: Model, States, Transitions, Property, Marginal, then num_configs for data
    num_configs = len(configs)
    col_spec = "lrrrr" + "".join(
        ["|" + "".join("r" for _ in arithmetic_modes) for _ in methods]
    )  # Model (left), States (right), Transitions (right), Property (right), Marginal (right), then data columns

    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header: method names spanning columns
    header1 = "Model & States & Trans. & Prop & Marg."
    method_spans = {}
    for method in methods:
        count = sum(1 for m, a in configs if m == method)
        method_spans[method] = count

    for i, method in enumerate(methods):
        span = method_spans[method]
        vline = "|" if i != len(methods) - 1 else ""
        if span > 1:
            header1 += f" & \\multicolumn{{{span}}}{{c{vline}}}{{{method}}}"
        else:
            header1 += f" & {method}"
    header1 += " \\\\"
    lines.append(header1)

    # Header: arithmetic mode names
    header2 = " & & & &"
    for method in methods:
        for arith in arithmetic_modes:
            if (method, arith) in configs:
                header2 += f" & {get_arithmetic_mode_name(arith)}"
    header2 += " \\\\"
    lines.append(header2)
    lines.append("\\midrule")

    # Data rows
    prev_source = None
    for model_idx, model in enumerate(models):
        if model not in properties_map:
            raise ValueError(f"Model {model} not found in properties_map")
        props = properties_map[model]
        if not props:
            raise ValueError(f"Model {model} has no properties")

        # Add horizontal line between different sources
        current_source = model_sources[model]
        if prev_source is not None and current_source != prev_source:
            lines.append("\\midrule")
        prev_source = current_source

        for i, prop in enumerate(props):
            # Model name, states, and transitions only on first row for this model
            if i == 0:
                # Get display name for model (with LaTeX escaping)
                model_tex = get_model_name(model)
                lines.append(f"\\multirow{{{len(props)}}}{{*}}{{{model_tex}}}")

                # Add states and transitions
                if model in model_stats:
                    states, transitions = model_stats[model]
                    lines[
                        -1
                    ] += f" & \\multirow{{{len(props)}}}{{*}}{{{states}}} & \\multirow{{{len(props)}}}{{*}}{{{transitions}}}"
                else:
                    lines[
                        -1
                    ] += f" & \\multirow{{{len(props)}}}{{*}}{{---}} & \\multirow{{{len(props)}}}{{*}}{{---}}"
            else:
                lines.append(" & &")

            # Property number (1-indexed) and marginal
            marginal = property_marginals.get((model, prop))
            if marginal is not None:
                # Format marginal to 3 decimal places
                marginal_str = f"{marginal:.3f}"
            else:
                marginal_str = "---"
            row = f" & {i+1} & {marginal_str}"

            # Find best time for this row (excluding timeouts and incorrect results)
            best_time = min(
                (t
                for t, c, qc, to in data[model][prop].values()
                if c and qc and not to and t is not None),
                default=None,
            )

            # Add data cells
            for config in configs:
                if config in data[model][prop]:
                    time_val, is_correct, is_quan_correct, is_timeout = data[model][
                        prop
                    ][config]

                    if is_timeout:
                        cell = "TO"
                    elif is_correct is False:
                        cell = "$\\times$"
                    elif is_quan_correct is False:
                        cell = "$\\dagger$"
                    elif time_val is None:
                        cell = "---"
                    else:
                        # Format time with at most 3 decimal places
                        cell = f"{time_val:.3f}"
                        # Remove trailing zeros after decimal point
                        if "." in cell:
                            cell = cell.rstrip("0").rstrip(".")

                        # Bold if this is the best time
                        if best_time is not None and abs(time_val - best_time) < 1e-9:
                            cell = f"\\textbf{{{cell}}}"
                else:
                    cell = "---"

                row += f" & {cell}"

            row += " \\\\"
            lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated LaTeX table: {output_file}")


def create_scatter_plot(
    points,
    color_map,
    markers,
    marker_labels,
    xlabel,
    ylabel,
    title,
    filename,
    output_dir
):
    """Abstract scatter plot creation with error and timeout handling.

    Args:
        points: List of tuples (val1, val2, model, qtype, is_correct1, is_correct2, is_timeout1, is_timeout2)
        color_map: Dictionary mapping model names to colors
        markers: Dictionary mapping query types to marker symbols
        marker_labels: Dictionary mapping query types to display labels
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename
        output_dir: Output directory path
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Find max value to place incorrect points on a line above
    # max_val = max(
    #     max(
    #         (v1 for v1, _, _, _, c1, _, to1, _ in points if c1 and not to1),
    #         default=1.0,
    #     ),
    #     max(
    #         (v2 for _, v2, _, _, _, c2, _, to2 in points if c2 and not to2),
    #         default=1.0,
    #     ),
    # )
    max_val = 600
    error_line = max_val * 5  # Place error line 5x higher
    timeout_line = max_val * 10  # Place timeout line 10x higher

    # Plot points
    labeled_models = set()
    for v1, v2, model, qtype, c1, c2, to1, to2 in points:
        label = get_model_name(model) if model not in labeled_models else None
        if label:
            labeled_models.add(model)
        # Timeouts go to timeout_line, wrong results go to error_line

        if to1:
            v1 = timeout_line
        elif c1 is None:
            pass
        elif not c1:
            v1 = error_line

        if to2:
            v2 = timeout_line
        elif c2 is None:
            pass
        elif not c2:
            v2 = error_line

        ax.scatter(
            v1,
            v2,
            color=color_map[model],
            marker=markers[qtype],
            s=110,
            alpha=0.75,
            edgecolors="k",
            linewidths=0.4,
            label=label,
        )

    # Draw horizontal error line and timeout line
    ax.axvline(x=error_line, color="k", linestyle="--", alpha=0.3, linewidth=1)
    ax.axhline(y=error_line, color="k", linestyle="--", alpha=0.3, linewidth=1)
    ax.axvline(x=timeout_line, color="orange", linestyle=":", alpha=0.4, linewidth=1.5)
    ax.axhline(y=timeout_line, color="orange", linestyle=":", alpha=0.4, linewidth=1.5)

    # Reference lines
    min_val = min(
        min([100] + [v1 for v1, _, _, _, c1, _, to1, _ in points if c1 and not to1]),
        min([100] + [v2 for _, v2, _, _, _, c2, _, to2 in points if c2 and not to2]),
    )
    min_stop_lines = min_val * 0.1

    ax.plot(
        [min_stop_lines, error_line],
        [min_stop_lines, error_line],
        "k-",
        linewidth=1,
        label="1:1",
    )
    ax.plot(
        [min_stop_lines / 10, error_line / 10],
        [min_stop_lines, error_line],
        "k--",
        linewidth=0.6,
        alpha=0.5,
        label="10x",
    )
    ax.plot(
        [min_stop_lines, error_line],
        [min_stop_lines / 10, error_line / 10],
        "k--",
        linewidth=0.6,
        alpha=0.5,
    )
    ax.plot(
        [min_stop_lines / 100, error_line / 100],
        [min_stop_lines, error_line],
        "k:",
        linewidth=0.6,
        alpha=0.5,
        label="100x",
    )
    ax.plot(
        [min_stop_lines, error_line],
        [min_stop_lines / 100, error_line / 100],
        "k:",
        linewidth=0.6,
        alpha=0.5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Build legend
    model_handles, model_labels = ax.get_legend_handles_labels()
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[qt],
            color="k",
            linestyle="",
            markerfacecolor="w",
            markeredgecolor="k",
            markersize=9,
            label=marker_labels[qt],
        )
        for qt in markers.keys()
    ]

    ax.legend(
        model_handles + marker_handles,
        model_labels + [h.get_label() for h in marker_handles],
        title="Model",
        framealpha=0.9,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    # Add tick labels for incorrect results on the error line and timeout line
    yticks = list(t for t in ax.get_yticks() if t < error_line)
    yticks.extend([error_line, timeout_line])
    ax.set_yticks(yticks)
    ax.set_yticks([t for t in ax.get_yticks(minor=True) if t < error_line], minor=True)
    tick_labels = [
        (
            rf"$10^{{{int(np.log10(t))}}}$"
            if t not in [error_line, timeout_line]
            else (r"$\times$" if t == error_line else r"$\mathcal{T}$")
        )
        for t in ax.get_yticks()
    ]
    ax.set_yticklabels(tick_labels)

    xticks = list(t for t in ax.get_xticks() if t < error_line)
    xticks.extend([error_line, timeout_line])
    ax.set_xticks(xticks)
    ax.set_xticks([t for t in ax.get_xticks(minor=True) if t < error_line], minor=True)
    tick_labels = [
        (
            rf"$10^{{{int(np.log10(t))}}}$"
            if t not in [error_line, timeout_line]
            else (r"$\times$" if t == error_line else r"$\mathcal{T}$")
        )
        for t in ax.get_xticks()
    ]
    ax.set_xticklabels(tick_labels)

    # Set limits, ensuring they're positive for log scale
    left_lim = max(min_val * 0.5, 1e-6)
    right_lim = timeout_line * 2
    bottom_lim = max(min_val * 0.5, 1e-6)
    top_lim = timeout_line * 2

    ax.set_xlim(left=left_lim, right=right_lim)
    ax.set_ylim(bottom=bottom_lim, top=top_lim)

    plt.tight_layout()
    plt.savefig(output_dir / filename, backend="pgf")
    plt.close()
    print(f"Saved: {output_dir / filename}")


def plot_method_scatter(results, output_dir, correctness):
    """Create scatter plots for all method×arithmetic combinations.

    Incorrect results are placed on a line above all other points.
    """
    from itertools import combinations

    # Include both successful results and timeouts
    plottable = [r for r in results if r["success"] or r["timeout"]]

    if not plottable:
        print("No results with arithmetic_mode to plot")
        return

    # Get all method×arithmetic combinations
    combinations_set = sorted(
        set((r["method"], r["arithmetic_mode"]) for r in plottable)
    )

    baselines = [combo for combo in combinations_set if combo[0] == "restart"]

    if len(combinations_set) < 2:
        print(
            f"Scatter plot requires at least 2 method×arithmetic combinations, found {len(combinations_set)}"
        )
        return

    # Create scatter plots for all combination pairs
    combo_pairs = list(product(combinations_set, baselines))

    for (method1, arith1), (method2, arith2) in combo_pairs:
        if (method1, arith1) == (method2, arith2):
            continue  # Skip comparing method to itself

        combo1_name = f"{method1}/{get_arithmetic_mode_name(arith1)}"
        combo2_name = f"{method2}/{get_arithmetic_mode_name(arith2)}"

        combo1_results = [
            r
            for r in plottable
            if r["method"] == method1 and r["arithmetic_mode"] == arith1
        ]
        combo2_results = [
            r
            for r in plottable
            if r["method"] == method2 and r["arithmetic_mode"] == arith2
        ]

        if not combo1_results or not combo2_results:
            raise ValueError(
                f"Missing results for comparison: {combo1_name} has {len(combo1_results)} results, {combo2_name} has {len(combo2_results)} results"
            )

        # Extract properties from results for these combinations
        properties_map = get_properties_from_results(combo1_results + combo2_results)

        # Collect per-model colors
        all_results = combo1_results + combo2_results
        models = sorted(
            set(
                r["model"]
                for r in all_results
                if not exclude_model(r["model"], "scatter")
            )
        )
        color_map = get_model_colors(models)

        query_types = ["quantitative", "bounded"]
        markers = {"quantitative": "o", "bounded": "^"}
        marker_labels = {qt: get_query_type_name(qt).capitalize() for qt in query_types}

        # Collect points for this combination pair
        points = (
            []
        )  # (t1, t2, model, qtype, is_correct1, is_correct2, is_timeout1, is_timeout2)
        for model in models:
            if model not in properties_map:
                raise ValueError(
                    f"Model {model} not found in properties_map for scatter plot"
                )
            for qtype in query_types:
                for path_formula in properties_map[model]:
                    r1_list = [
                        r
                        for r in combo1_results
                        if r["model"] == model
                        and r["query_type"] == qtype
                        and r["path_formula"] == path_formula
                    ]
                    r2_list = [
                        r
                        for r in combo2_results
                        if r["model"] == model
                        and r["query_type"] == qtype
                        and r["path_formula"] == path_formula
                    ]
                    if r1_list and r2_list:
                        key1 = (model, method1, arith1, qtype, path_formula)
                        key2 = (model, method2, arith2, qtype, path_formula)
                        is_correct1 = key1 in correctness and correctness[key1]
                        is_correct2 = key2 in correctness and correctness[key2]
                        is_timeout1 = r1_list[0]["timeout"]
                        is_timeout2 = r2_list[0]["timeout"]
                        points.append(
                            (
                                r1_list[0]["time"],
                                r2_list[0]["time"],
                                model,
                                qtype,
                                is_correct1,
                                is_correct2,
                                is_timeout1,
                                is_timeout2,
                            )
                        )

        if not points:
            raise ValueError(
                f"No data points found for scatter plot: {combo1_name} vs {combo2_name}"
            )

        # Sanitize names for filename (remove LaTeX symbols)
        combo1_file = (
            combo1_name.replace("/", "_")
            .replace("$", "")
            .replace("\\", "")
            .replace("varepsilon", "eps")
        )
        combo2_file = (
            combo2_name.replace("/", "_")
            .replace("$", "")
            .replace("\\", "")
            .replace("varepsilon", "eps")
        )
        filename = f"scatter_{combo1_file}_vs_{combo2_file}.pdf"

        create_scatter_plot(
            points=points,
            color_map=color_map,
            markers=markers,
            marker_labels=marker_labels,
            xlabel=f"Time for {combo1_name} (s)",
            ylabel=f"Time for {combo2_name} (s)",
            title=f"Method Comparison: {combo1_name} vs {combo2_name}",
            filename=filename,
            output_dir=output_dir,
        )


def plot_iterations_scatter(results, output_dir, correctness):
    """Create scatter plots comparing iterations between bisection and bisection-advanced for same arithmetic modes."""

    # Include both successful results and timeouts
    plottable = [
        r for r in results if (r["success"] or r["timeout"]) and "iterations" in r
    ]

    if not plottable:
        print("No results with iterations data to plot")
        return

    # Get all arithmetic modes
    arithmetic_modes = sorted(set(r["arithmetic_mode"] for r in plottable))

    # Filter to only bisection and bisection-advanced methods
    method1 = "bisection"
    method2 = "bisection-advanced"

    for arith in arithmetic_modes:
        arith_name = get_arithmetic_mode_name(arith)

        method1_results = [
            r
            for r in plottable
            if r["method"] == method1 and r["arithmetic_mode"] == arith
        ]
        method2_results = [
            r
            for r in plottable
            if r["method"] == method2 and r["arithmetic_mode"] == arith
        ]

        if not method1_results or not method2_results:
            continue

        # Extract properties from results for these methods
        properties_map = get_properties_from_results(method1_results + method2_results)

        # Collect per-model colors
        all_results = method1_results + method2_results
        models = sorted(
            set(
                r["model"]
                for r in all_results
                if not exclude_model(r["model"], "iterations_scatter")
            )
        )
        color_map = get_model_colors(models)

        query_types = ["quantitative"]
        markers = {"quantitative": "o"}
        marker_labels = {qt: get_query_type_name(qt).capitalize() for qt in query_types}

        # Collect points for this arithmetic mode
        points = []
        # (iter1, iter2, model, qtype, is_correct1, is_correct2, is_timeout1, is_timeout2)

        for model in models:
            if model not in properties_map:
                print(
                    f"Model {model} not found in properties_map for iterations scatter plot"
                )

            qtype = "quantitative"
            for path_formula in properties_map[model]:
                r1_list = [
                    r
                    for r in method1_results
                    if r["model"] == model
                    and r["query_type"] == qtype
                    and r["path_formula"] == path_formula
                ]
                r2_list = [
                    r
                    for r in method2_results
                    if r["model"] == model
                    and r["query_type"] == qtype
                    and r["path_formula"] == path_formula
                ]
                if r1_list and r2_list:
                    key1 = (model, method1, arith, qtype, path_formula)
                    key2 = (model, method2, arith, qtype, path_formula)
                    is_correct1 = key1 in correctness and correctness[key1]
                    is_correct2 = key2 in correctness and correctness[key2]
                    is_timeout1 = r1_list[0]["timeout"]
                    is_timeout2 = r2_list[0]["timeout"]

                    # Get iterations, default to 1 if not present, timeout, or None
                    iter1 = r1_list[0]["iterations"]
                    iter2 = r2_list[0]["iterations"]
                    iter1 = 1 if iter1 is None else iter1
                    iter2 = 1 if iter2 is None else iter2

                    # Ensure minimum value of 1 for log scale
                    iter1 = iter1 if not is_timeout1 else 1
                    iter2 = iter2 if not is_timeout2 else 1
                    assert iter1 is not None
                    assert iter2 is not None
                    points.append(
                        (
                            iter1,
                            iter2,
                            model,
                            qtype,
                            is_correct1,
                            is_correct2,
                            is_timeout1,
                            is_timeout2,
                        )
                    )

        if not points:
            continue

        # Sanitize names for filename
        arith_file = (
            arith_name.replace("/", "_")
            .replace("$", "")
            .replace("\\", "")
            .replace("varepsilon", "eps")
        )
        filename = f"scatter_iterations_{method1}_vs_{method2}_{arith_file}.pdf"

        create_scatter_plot(
            points=points,
            color_map=color_map,
            markers=markers,
            marker_labels=marker_labels,
            xlabel=f"Iterations for {method1}/{arith_name}",
            ylabel=f"Iterations for {method2}/{arith_name}",
            title=f"Iterations Comparison: {method1} vs {method2} ({arith_name})",
            filename=filename,
            output_dir=output_dir,
        )


def plot_speedup_heatmap(
    results,
    output_dir,
    correctness,
    query_type="quantitative",
    baseline=("restart", "exact"),
):
    """Plot heatmap showing speedup relative to exact restart baseline for all method×arithmetic combinations."""
    from matplotlib.colors import LogNorm

    # Filter to specific query type, include both successful and timeout results
    filtered_results = [
        r
        for r in results
        if r["query_type"] == query_type
        and (r["success"] or r["timeout"])
        and r["arithmetic_mode"]
    ]

    if not filtered_results:
        print(f"No {query_type} results to plot heatmap")
        return

    models = sorted(
        set(
            r["model"]
            for r in filtered_results
            if not exclude_model(r["model"], "heatmap")
        )
    )
    methods = sorted(set(r["method"] for r in filtered_results))
    arithmetic_modes = sorted(set(r["arithmetic_mode"] for r in filtered_results))

    # Extract properties from filtered results
    properties_map = get_properties_from_results(filtered_results)

    # Create method×arithmetic combinations (excluding exact restart which is baseline)
    combinations = []
    for mode in arithmetic_modes:
        for method in methods:
            combo = (method, mode, f"{method}/{get_arithmetic_mode_name(mode)}")
            if not (method == baseline[0] and mode == baseline[1]):
                combinations.append(combo)

    combinations.sort()

    if not combinations:
        print("No method/arithmetic combinations to compare")
        return

    # Build list of (model, property) columns
    columns = []
    model_boundaries = []  # Track where each model's columns start
    col_idx = 0
    for model in models:
        if model not in properties_map:
            raise ValueError(f"Model {model} not found in properties_map for heatmap")
        model_props = properties_map[model]
        model_boundaries.append((model, col_idx, col_idx + len(model_props)))
        for prop in model_props:
            columns.append((model, prop))
            col_idx += 1

    num_columns = len(columns)

    # Calculate speedups: baseline_time / method_time (baseline = exact restart)
    speedup_matrix = np.zeros((len(combinations), num_columns))
    wrong_answer_matrix = np.zeros((len(combinations), num_columns), dtype=bool)
    wrong_quantitative_answer_matrix = np.zeros(
        (len(combinations), num_columns), dtype=bool
    )
    timeout_matrix = np.zeros((len(combinations), num_columns), dtype=bool)
    baseline_timeout_matrix = np.zeros((len(combinations), num_columns), dtype=bool)
    missing_matrix = np.zeros((len(combinations), num_columns), dtype=bool)

    for i, (method, mode, combo_name) in enumerate(combinations):
        for j, (model, path_prop) in enumerate(columns):
            # Baseline is exact restart
            baseline_results = [
                r
                for r in filtered_results
                if r["model"] == model
                and r["method"] == baseline[0]
                and r["arithmetic_mode"] == baseline[1]
                and r["path_formula"] == path_prop
            ]
            # Current method/mode combination
            method_results = [
                r
                for r in filtered_results
                if r["model"] == model
                and r["method"] == method
                and r["arithmetic_mode"] == mode
                and r["path_formula"] == path_prop
            ]

            if baseline_results and method_results:
                baseline_time = baseline_results[0]["time"]
                method_time = method_results[0]["time"]
                key = (model, method, mode, query_type, path_prop)

                is_timeout = method_results[0]["timeout"]
                is_baseline_timeout = baseline_results[0]["timeout"]
                is_correct = key in correctness and (correctness[key] is None or correctness[key])
                quan_key = (model, method, mode, "quantitative", path_prop)
                is_quan_correct = not quan_key in correctness or (correctness[quan_key] is None or correctness[quan_key])

                timeout_matrix[i, j] = is_timeout
                baseline_timeout_matrix[i, j] = is_baseline_timeout
                wrong_answer_matrix[i, j] = not is_correct
                wrong_quantitative_answer_matrix[i, j] = not is_quan_correct

                if (
                    is_correct
                    and is_quan_correct
                    and not is_timeout
                    and not is_baseline_timeout
                    and method_time > 0
                ):
                    speedup_matrix[i, j] = baseline_time / method_time
                elif is_timeout or is_baseline_timeout or not is_correct:
                    speedup_matrix[i, j] = 0  # Will be masked
                else:
                    missing_matrix[i, j] = True
                    speedup_matrix[i, j] = 0
            else:
                missing_matrix[i, j] = True
                speedup_matrix[i, j] = 0

    # Mask zeros
    speedup_matrix = np.ma.masked_equal(speedup_matrix, 0)

    # Check if we have any valid data
    if isinstance(speedup_matrix.mask, np.ndarray):
        all_masked = speedup_matrix.mask.all()
    else:
        all_masked = speedup_matrix.mask

    if all_masked:
        print(
            f"No valid speedup data for {query_type} heatmap (all timeout/incorrect/missing)"
        )
        return

    fig_width = max(num_columns * 0.7, 8) + 3
    fig_height = len(combinations) * 0.5 + 2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Use LogNorm centered at 1.0 for better visualization
    # Values < 1 (slower) appear red, values > 1 (faster) appear green
    unmasked_data = speedup_matrix[~speedup_matrix.mask]
    if len(unmasked_data) == 0:
        print(f"No unmasked data for {query_type} heatmap")
        return

    data_min = np.min(unmasked_data)
    data_max = np.max(unmasked_data)

    # Calculate symmetric log range around 1.0
    log_range = max(abs(np.log10(data_min)), abs(np.log10(data_max)))
    vmin = 10 ** (-log_range)
    vmax = 10**log_range

    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn  # type: ignore

    im = ax.imshow(speedup_matrix, aspect="auto", cmap=cmap, norm=norm)

    # Set up column ticks and labels - show model names at group centers
    tick_positions = []
    tick_labels = []
    for model, start_col, end_col in model_boundaries:
        if end_col > start_col:
            center = (start_col + end_col - 1) / 2
            tick_positions.append(center)
            tick_labels.append(get_model_name(model))
            # Draw vertical lines to separate model groups
            if start_col > 0:
                ax.axvline(x=start_col - 0.5, color="k", linestyle="-", linewidth=1.5)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(combinations)))
    ax.set_yticklabels([combo_name for _, _, combo_name in combinations])

    # Add text annotations
    for i in range(len(combinations)):
        for j in range(num_columns):
            if isinstance(speedup_matrix.mask, np.ndarray):
                is_masked = speedup_matrix.mask[i, j]
            else:
                is_masked = speedup_matrix.mask

            if baseline_timeout_matrix[i, j]:
                ax.text(
                    j,
                    i,
                    "T(B)",
                    ha="center",
                    va="center",
                    color="orange",
                    fontsize=6,
                    weight="bold",
                )
            elif timeout_matrix[i, j]:
                ax.text(
                    j,
                    i,
                    "T(M)",
                    ha="center",
                    va="center",
                    color="orange",
                    fontsize=6,
                    weight="bold",
                )
            elif wrong_answer_matrix[i, j]:
                ax.text(
                    j,
                    i,
                    "✗",
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=9,
                    weight="bold",
                )
            elif wrong_quantitative_answer_matrix[i, j]:
                ax.text(
                    j,
                    i,
                    "✗Q",
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=6,
                    weight="bold",
                )
            elif missing_matrix[i, j]:
                pass  # Leave empty for missing data
            elif not is_masked:
                text = f"{speedup_matrix[i, j]:.1f}x"
                fontsize = 6 if num_columns > 20 else 7
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=fontsize,
                )

    ax.set_xlabel("Model × Property")
    ax.set_ylabel("Method/Arithmetic")
    query_label = (
        "Quantitative (Pmax=?)"
        if query_type == "quantitative"
        else "Qualitative (Pmax>=θ)"
    )
    ax.set_title(
        f"Speedup relative to {baseline[0]}/{baseline[1]} - {query_label}\n(baseline_time / method_time)"
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Speedup Factor")

    plt.tight_layout()
    filename = f"speedup_heatmap_{get_query_type_name(query_type)}.pdf"
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()
    print(f"Saved: {output_dir / filename}")


def plot_speedup_vs_marginal(
    results,
    output_dir,
    correctness,
    baseline_combo=("restart", "exact"),
    target_combo=None,
):
    """Plot speedup vs marginal probability comparing two configs.

    Shows all properties and query types in one plot.
    X-axis: marginal probability, Y-axis: speedup
    Different markers for quantitative (o) and bounded (^) queries.
    Incorrect results placed on a line above all points.

    Args:
        results: Benchmark results
        output_dir: Output directory for plots
        correctness: Correctness dict from validate_results
        baseline_combo: (method, arithmetic_mode) tuple for baseline (default: restart/exact)
        target_combo: (method, arithmetic_mode) tuple to compare (default: all others vs baseline)
    """
    # Get all method×arithmetic combinations, include timeouts
    all_combos = sorted(
        set(
            (r["method"], r["arithmetic_mode"])
            for r in results
            if (r.get("success") or r.get("timeout"))
        )
    )

    if baseline_combo not in all_combos:
        print(f"Baseline {baseline_combo} not found in results")
        return

    # If target_combo specified, create single plot for that pair
    if target_combo:
        combos_to_plot = [(baseline_combo, target_combo)]
    else:
        # Create plots for all combinations vs baseline
        combos_to_plot = [
            (baseline_combo, c) for c in all_combos if c != baseline_combo
        ]

    for base_combo, tgt_combo in combos_to_plot:
        base_method, base_arith = base_combo
        tgt_method, tgt_arith = tgt_combo
        base_name = f"{base_method}/{get_arithmetic_mode_name(base_arith)}"
        tgt_name = f"{tgt_method}/{get_arithmetic_mode_name(tgt_arith)}"

        # Get results for both configurations, include timeouts
        baseline_results = [
            r
            for r in results
            if (r["success"] or r["timeout"])
            and r["method"] == base_method
            and r["arithmetic_mode"] == base_arith
        ]
        target_results = [
            r
            for r in results
            if (r["success"] or r["timeout"])
            and r["method"] == tgt_method
            and r["arithmetic_mode"] == tgt_arith
        ]

        if not baseline_results or not target_results:
            raise ValueError(
                f"Missing results for comparison: {base_name} has {len(baseline_results)} results, {tgt_name} has {len(target_results)} results"
            )

        # Extract properties from these results
        properties_map = get_properties_from_results(baseline_results + target_results)

        # Collect color map for models
        models = sorted(
            set(
                r["model"]
                for r in baseline_results + target_results
                if not exclude_model(r["model"], "marginal")
            )
        )
        color_map = get_model_colors(models)

        # Markers for query types
        markers = {"quantitative": "o", "bounded": "^"}
        query_types = ["quantitative", "bounded"]
        marker_labels = {qt: get_query_type_name(qt).capitalize() for qt in query_types}

        # Collect all data points (model, property, query_type, marginal, speedup, is_correct)
        points = []
        for model in models:
            # if not model.startswith("brp-"):
            #     continue
            if model not in properties_map:
                raise ValueError(
                    f"Model {model} not found in properties_map for speedup_vs_marginal"
                )
            for path_formula in properties_map[model]:
                for query_type in query_types:
                    baseline = [
                        r
                        for r in baseline_results
                        if r["model"] == model
                        and r["path_formula"] == path_formula
                        and r["query_type"] == query_type
                    ]
                    target = [
                        r
                        for r in target_results
                        if r["model"] == model
                        and r["path_formula"] == path_formula
                        and r["query_type"] == query_type
                    ]

                    if baseline and target:
                        # Use baseline marginal since target may timeout and lack marginal
                        marginal = (
                            baseline[0]["marginal"]
                            if "marginal" in baseline[0]
                            else baseline[0]["value"]
                        )
                        # Raise exception if no marginal available
                        if marginal is None:
                            print(
                                f"WARNING: No marginal for model={model}, property={path_formula}, query_type={query_type}"
                            )
                            continue

                        if target[0]["timeout"]:
                            speedup = 0
                        else:
                            speedup = (
                                baseline[0]["time"] / target[0]["time"]
                                if target[0]["time"] > 0
                                else 0
                            )

                        # Clamp to positive for log scale
                        speedup = max(speedup, 1e-12)

                        is_timeout = target[0]["timeout"] or baseline[0]["timeout"]

                        if not is_timeout:
                            key_base = (
                                model,
                                base_method,
                                base_arith,
                                query_type,
                                path_formula,
                            )
                            key_tgt = (
                                model,
                                tgt_method,
                                tgt_arith,
                                query_type,
                                path_formula,
                            )
                            if key_tgt not in correctness:
                                raise ValueError(
                                    f"Correctness not found for key: model={model}, method={tgt_method}, arith={tgt_arith}, query_type={query_type}, path_formula={path_formula}"
                                )
                            is_correct = correctness[key_tgt] and correctness[key_base]
                        else:
                            is_correct = True

                        points.append(
                            (
                                marginal,
                                speedup,
                                model,
                                query_type,
                                is_correct,
                                is_timeout,
                                path_formula,
                            )
                        )

        if not points:
            raise ValueError(f"No valid data points for {base_name} vs {tgt_name}")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Determine error line (bottom) and timeout line (above error) and y-limits
        correct_speedups = [s for _, s, _, _, c, t, _ in points if c and not t]
        all_speedups = [s for _, s, _, _, _, _, _ in points]
        if correct_speedups:
            min_speedup = min(correct_speedups)
            max_speedup = max(correct_speedups)
        else:
            min_speedup = min(all_speedups) if all_speedups else 1.0
            max_speedup = max(all_speedups) if all_speedups else 1.0

        error_line_y = min(0.1, max(min_speedup / 5.0, 1e-12))
        timeout_line_y = error_line_y / 2.0
        y_min = timeout_line_y / 2.0
        y_max = max(max_speedup * 2.0, error_line_y * 2.0)

        # Track which models/query types we've already labeled
        labeled = set()

        # Plot points; timeouts go to timeout line, incorrect results go to error line at bottom
        for (
            marginal,
            speedup,
            model,
            qtype,
            is_correct,
            is_timeout,
            path_formula,
        ) in points:
            label_key = (model, qtype)
            label = (
                f"{get_model_name(model)}-{qtype[0]}"
                if label_key not in labeled
                else None
            )
            if label:
                labeled.add(label_key)

            if is_timeout:
                y_val = timeout_line_y
            elif is_correct:
                y_val = speedup
            else:
                y_val = error_line_y

            ax.scatter(
                marginal,
                y_val,
                color=color_map[model],
                marker=markers[qtype],
                s=100,
                alpha=0.7,
                edgecolors="k",
                linewidths=0.5,
                label=label,
            )

        # Horizontal guides: no-speedup line, error line, and timeout line at bottom
        ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(y=error_line_y, color="k", linestyle="--", alpha=0.3, linewidth=1)
        ax.axhline(
            y=timeout_line_y, color="orange", linestyle=":", alpha=0.4, linewidth=1.5
        )

        # ax.set_xscale("log")
        ax.set_xlabel("Marginal Probability")
        ax.set_ylabel(
            f"Speedup ({base_name.replace('/', '_')}/{tgt_name.replace('/', '_')})"
        )
        ax.set_yscale("log")
        ax.set_title(
            f"Speedup vs Marginal: {base_name} vs {tgt_name}\n(all properties and query types)"
        )
        ax.grid(True, alpha=0.3)

        # Build legend
        model_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[m],
                markersize=8,
                markeredgecolor="k",
                markeredgewidth=0.5,
                label=get_model_name(m),
            )
            for m in models
        ]

        qtype_handles = [
            Line2D(
                [0],
                [0],
                marker=markers[qt],
                color="k",
                linestyle="",
                markersize=9,
                label=marker_labels[qt],
            )
            for qt in query_types
        ]

        model_handles.append(
            Line2D(
                [0],
                [0],
                color="k",
                linestyle="--",
                label="No Speedup (y=1)",
            )
        )

        ax.legend(
            handles=model_handles + qtype_handles,
            title="Model / Query Type",
            framealpha=0.9,
            ncol=2,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        # Y ticks: include error line at bottom with × label
        yticks = [t for t in ax.get_yticks() if t > error_line_y]
        yticks.append(error_line_y)
        ax.set_yticks(yticks)
        ax.set_yticks(
            [t for t in ax.get_yticks(minor=True) if t > error_line_y], minor=True
        )
        tick_labels = [
            rf"$10^{{{int(np.log10(t))}}}$" if t != error_line_y else r"$\times$"
            for t in ax.get_yticks()
        ]
        ax.set_yticklabels(tick_labels)

        ax.set_ylim(bottom=y_min, top=y_max)

        plt.tight_layout()
        # Sanitize names for filename (remove LaTeX symbols)
        base_file = (
            base_name.replace("/", "_")
            .replace("$", "")
            .replace("\\", "")
            .replace("varepsilon", "eps")
        )
        tgt_file = (
            tgt_name.replace("/", "_")
            .replace("$", "")
            .replace("\\", "")
            .replace("varepsilon", "eps")
        )
        filename = f"speedup_vs_marginal_{base_file}_vs_{tgt_file}.pdf"
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()
        print(f"Saved: {output_dir / filename}")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("json_file", type=Path, help="JSON file with benchmark results")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for plots (default: plots/)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(args.json_file).parent / "plots"

    if not args.json_file.exists():
        print(f"Error: {args.json_file} does not exist")
        return 1

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(args.json_file)
    print(f"Loaded {len(results)} results from {args.json_file}")

    # Validate results and detect wrong answers
    print("\nValidating results...")
    correctness, wrong_by = validate_results(results)
    wrong_count = sum(1 for v in correctness.values() if not v)
    print(f"Found {wrong_count} wrong answers out of {len(correctness)} results")
    if wrong_count > 0:
        print("Wrong answers details:")
        for key, diff in wrong_by.items():
            model, method, arith, qtype, path_formula = key
            print(
                f"  Model: {model}, Method: {method}, Arithmetic: {arith}, Query: {qtype}, Path: {path_formula}, Wrong by: {diff}"
            )

    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    generate_latex_table(
        results,
        args.output / "runtime_table_quantitative.tex",
        correctness,
        query_type="quantitative",
    )
    generate_latex_table(
        results,
        args.output / "runtime_table_bounded.tex",
        correctness,
        query_type="bounded",
    )

    # Generate plots
    print("\nGenerating plots...")
    plot_speedup_heatmap(results, args.output, correctness, query_type="quantitative")
    plot_speedup_heatmap(results, args.output, correctness, query_type="bounded")
    plot_iterations_scatter(results, args.output, correctness)
    plot_speedup_vs_marginal(results, args.output, correctness)
    plot_method_scatter(results, args.output, correctness)

    print(f"\nAll plots and tables saved to {args.output}/")


if __name__ == "__main__":
    main()
