#!/usr/bin/env python3
"""
Plot benchmark results from JSON file.
"""

import argparse
from ast import mod
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from benchmark import PROPERTIES


def get_properties_from_results(results):
    """Extract properties from results data, grouped by model."""
    model_props = {}
    for r in results:
        model = r.get("model")
        path_formula = r.get("path_formula")
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
    allowed_brp = [
        "brp-N=16-MAX=8-PCHAN=0.010",
        "brp-N=32-MAX=9-PCHAN=0.010",
        "brp-N=64-MAX=10-PCHAN=0.010",
    ]

    if plot_type == "heatmap":
        if model_name.startswith("brp-") and model_name not in allowed_brp:
            return True

    return False


def load_results(json_file):
    """Load benchmark results from JSON file."""
    with open(json_file, "r") as f:
        return list(r for r in json.load(f) if r is not None)


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
        if not r.get("success") or r.get("timeout"):
            continue
        key = (r["model"], r["query_type"], r["path_formula"])
        if key not in by_model_query:
            by_model_query[key] = []
        by_model_query[key].append(r)

    for (model, query_type, path_formula), group_results in by_model_query.items():
        if query_type == "quantitative":
            # For quantitative queries, use majority voting on exact arithmetic values
            exact_results = [
                r for r in group_results if r.get("arithmetic_mode") == "force-exact"
            ]

            if not exact_results:
                print(
                    f"No force-exact results for model {model}, query {query_type}, path_formula {path_formula}, skipping validation"
                )
                continue

            # Get majority value (round to 10 decimal places for comparison)
            exact_values = [round(r["value"], 10) for r in exact_results]
            value_counts = Counter(exact_values)
            majority_value, count = value_counts.most_common(1)[0]
            if count < 2:
                print(
                    f"No majority value for model {model}, query {query_type}, path_formula {path_formula}, skipping validation"
                )

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

                if r["arithmetic_mode"] == "force-exact":
                    correctness[result_key] = rounded_value == majority_value
                    if not correctness[result_key]:
                        # This is very suspect, as exact arithmetic should agree
                        print(
                            f"Discrepancy in force-exact arithmetic for {result_key}: value {rounded_value} vs majority {majority_value}. All force-exact results: { {r['method']: r['value'] for r in exact_results} }"
                        )
                else:  # float or imprecise exact
                    # Float values should be close to majority (within 1e-5)
                    correctness[result_key] = abs(r["value"] - majority_value) < 1e-5

                if not correctness[result_key]:
                    wrong_by[result_key] = abs(r["value"] - majority_value)

        elif query_type == "bounded":
            # For bounded queries, check if they match the threshold comparison
            # First get the correct quantitative value for this model
            quant_results = by_model_query.get(
                (model, "quantitative", path_formula), []
            )
            exact_quant = [
                r for r in quant_results if r.get("arithmetic_mode") == "force-exact"
            ]

            if not exact_quant:
                continue

            # Get majority quantitative value
            exact_values = [round(r["value"], 10) for r in exact_quant]
            value_counts = Counter(exact_values)
            majority_quant_value, _ = value_counts.most_common(1)[0]

            # For each bounded result, check if it matches the expected boolean
            for r in group_results:
                result_key = (
                    r["model"],
                    r["method"],
                    r["arithmetic_mode"],
                    r["query_type"],
                    r["path_formula"],
                )
                threshold = r.get("threshold", 0.5)
                expected_value = 1.0 if majority_quant_value >= threshold else 0.0
                correctness[result_key] = r["value"] == expected_value

                if not correctness[result_key]:
                    wrong_by[result_key] = abs(r["value"] - expected_value)

    return correctness, wrong_by


def plot_method_scatter(results, output_dir, correctness):
    """Create scatter plots for all method×arithmetic combinations.

    Incorrect results are placed on a line above all other points.
    """
    from itertools import combinations

    # Include both successful results and timeouts
    plottable = [r for r in results if r.get("success") or r.get("timeout")]

    if not plottable:
        print("No results with arithmetic_mode to plot")
        return

    # Get all method×arithmetic combinations
    combinations_set = sorted(
        set((r["method"], r["arithmetic_mode"]) for r in plottable)
    )

    if len(combinations_set) < 2:
        print(
            f"Scatter plot requires at least 2 method×arithmetic combinations, found {len(combinations_set)}"
        )
        return

    # Create scatter plots for all combination pairs
    combo_pairs = list(combinations(combinations_set, 2))

    for (method1, arith1), (method2, arith2) in combo_pairs:
        combo1_name = f"{method1}/{arith1}"
        combo2_name = f"{method2}/{arith2}"

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
            continue

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
        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        color_map = {m: palette[i % len(palette)] for i, m in enumerate(models)}

        query_types = ["quantitative", "bounded"]
        markers = {"quantitative": "o", "bounded": "^"}

        # Collect points for this combination pair
        points = []  # (t1, t2, model, qtype, is_correct1, is_correct2)
        for model in models:
            for qtype in query_types:
                for path_formula in properties_map.get(model, []):
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
                        is_correct1 = correctness.get(key1, True)
                        is_correct2 = correctness.get(key2, True)
                        is_timeout1 = r1_list[0].get("timeout", False)
                        is_timeout2 = r2_list[0].get("timeout", False)
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
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        # Find max time to place incorrect points on a line above
        max_time = max(
            max((t1 for t1, _, _, _, c1, _, _, _ in points if c1), default=1.0),
            max((t2 for _, t2, _, _, _, c2, _, _ in points if c2), default=1.0),
        )
        error_line = max_time * 5  # Place error line 5x higher
        timeout_line = max_time * 10  # Place timeout line 10x higher

        # Plot correct points
        labeled_models = set()
        for t1, t2, model, qtype, c1, c2, to1, to2 in points:
            label = model if model not in labeled_models else None
            if label:
                labeled_models.add(model)
            # Timeouts go to timeout_line, wrong results go to error_line
            if to1:
                t1 = timeout_line
            elif not c1:
                t1 = error_line
            if to2:
                t2 = timeout_line
            elif not c2:
                t2 = error_line

            # Use normal marker for all points
            ax.scatter(
                t1,
                t2,
                c=color_map[model],
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
        ax.axvline(
            x=timeout_line, color="orange", linestyle=":", alpha=0.4, linewidth=1.5
        )
        ax.axhline(
            y=timeout_line, color="orange", linestyle=":", alpha=0.4, linewidth=1.5
        )

        # Reference lines
        min_time = min(
            min(t1 for t1, _, _, _, _, _, _, _ in points),
            min(t2 for _, t2, _, _, _, _, _, _ in points),
        )
        min_stop_lines = min_time * 0.1

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
        ax.set_xlabel(f"Time for {combo1_name} (s)")
        ax.set_ylabel(f"Time for {combo2_name} (s)")
        ax.set_title(f"Method Comparison: {combo1_name} vs {combo2_name}")
        # ax.grid(True, alpha=0.3)

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
                label=f"{qt.capitalize()}",
            )
            for qt in query_types
        ]

        ax.legend(
            model_handles + marker_handles,
            model_labels + [h.get_label() for h in marker_handles],
            title="Model / Query Type",
            framealpha=0.9,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        # Add tick labels for incorrect results on the error line and timeout line

        yticks = list(t for t in ax.get_yticks() if t < error_line)
        yticks.extend([error_line, timeout_line])
        ax.set_yticks(yticks)
        ax.set_yticks(
            [t for t in ax.get_yticks(minor=True) if t < error_line], minor=True
        )
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
        ax.set_xticks(
            [t for t in ax.get_xticks(minor=True) if t < error_line], minor=True
        )
        tick_labels = [
            (
                rf"$10^{{{int(np.log10(t))}}}$"
                if t not in [error_line, timeout_line]
                else (r"$\times$" if t == error_line else r"$\mathcal{T}$")
            )
            for t in ax.get_xticks()
        ]
        ax.set_xticklabels(tick_labels)

        ax.set_xlim(left=min_time * 0.5, right=timeout_line * 2)
        ax.set_ylim(bottom=min_time * 0.5, top=timeout_line * 2)

        plt.tight_layout()
        filename = f"scatter_{combo1_name.replace('/', '_')}_vs_{combo2_name.replace('/', '_')}.pdf"
        plt.savefig(output_dir / filename, backend="pgf")
        plt.close()
        print(f"Saved: {output_dir / filename}")


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
        if r.get("query_type") == query_type
        and (r.get("success") or r.get("timeout"))
        and r.get("arithmetic_mode")
    ]

    if not filtered_results:
        print(f"No {query_type} results with arithmetic_mode to plot heatmap")
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
            combo = f"{method}/{mode}"
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
        model_props = properties_map.get(model, [])
        model_boundaries.append((model, col_idx, col_idx + len(model_props)))
        for prop in model_props:
            columns.append((model, prop))
            col_idx += 1

    num_columns = len(columns)

    # Calculate speedups: baseline_time / method_time (baseline = exact restart)
    speedup_matrix = np.zeros((len(combinations), num_columns))
    wrong_answer_matrix = np.zeros((len(combinations), num_columns), dtype=bool)
    timeout_matrix = np.zeros((len(combinations), num_columns), dtype=bool)
    baseline_timeout_matrix = np.zeros((len(combinations), num_columns), dtype=bool)
    missing_matrix = np.zeros((len(combinations), num_columns), dtype=bool)

    for i, combo in enumerate(combinations):
        method, mode = combo.split("/")
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

                is_timeout = method_results[0].get("timeout", False)
                is_baseline_timeout = baseline_results[0].get("timeout", False)
                is_correct = correctness.get(key, True)

                timeout_matrix[i, j] = is_timeout
                baseline_timeout_matrix[i, j] = is_baseline_timeout
                wrong_answer_matrix[i, j] = not is_correct

                if (
                    method_time > 0
                    and is_correct
                    and not is_timeout
                    and not is_baseline_timeout
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

    fig_width = max(num_columns * 0.4, 8) + 3
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
    cmap = plt.cm.RdYlGn

    im = ax.imshow(speedup_matrix, aspect="auto", cmap=cmap, norm=norm)

    # Set up column ticks and labels - show model names at group centers
    tick_positions = []
    tick_labels = []
    for model, start_col, end_col in model_boundaries:
        if end_col > start_col:
            center = (start_col + end_col - 1) / 2
            tick_positions.append(center)
            tick_labels.append(model)
            # Draw vertical lines to separate model groups
            if start_col > 0:
                ax.axvline(x=start_col - 0.5, color="k", linestyle="-", linewidth=1.5)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(combinations)))
    ax.set_yticklabels(combinations)

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
        "Quantitative (Pmax=?)" if query_type == "quantitative" else "Bounded (Pmax>=θ)"
    )
    ax.set_title(
        f"Speedup relative to {baseline[0]}/{baseline[1]} - {query_label}\n(baseline_time / method_time)"
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Speedup Factor")

    plt.tight_layout()
    filename = f"speedup_heatmap_{query_type}.pdf"
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
        base_name = f"{base_method}/{base_arith}"
        tgt_name = f"{tgt_method}/{tgt_arith}"

        # Get results for both configurations, include timeouts
        baseline_results = [
            r
            for r in results
            if (r.get("success") or r.get("timeout"))
            and r["method"] == base_method
            and r["arithmetic_mode"] == base_arith
        ]
        target_results = [
            r
            for r in results
            if (r.get("success") or r.get("timeout"))
            and r["method"] == tgt_method
            and r["arithmetic_mode"] == tgt_arith
        ]

        if not baseline_results or not target_results:
            continue

        # Extract properties from these results
        properties_map = get_properties_from_results(baseline_results + target_results)

        # Collect color map for models
        models = sorted(
            set(
                r["model"]
                for r in baseline_results + target_results
                if not exclude_model(r["model"], "speedup_vs_marginal")
            )
        )
        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        color_map = {m: palette[i % len(palette)] for i, m in enumerate(models)}

        # Markers for query types
        markers = {"quantitative": "o", "bounded": "^"}
        query_types = ["quantitative", "bounded"]

        # Collect all data points (model, property, query_type, marginal, speedup, is_correct)
        points = []
        for model in models:
            for path_formula in properties_map.get(model, []):
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
                        marginal = baseline[0].get(
                            "marginal", baseline[0].get("value", 0)
                        )
                        # Skip if no marginal available
                        if marginal is None or marginal == 0:
                            continue

                        speedup = (
                            baseline[0]["time"] / target[0]["time"]
                            if target[0]["time"] > 0
                            else 0
                        )
                        # Clamp to positive for log scale
                        speedup = max(speedup, 1e-12)

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
                            continue
                        is_correct = correctness[key_tgt]
                        is_timeout = target[0].get("timeout", False)

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
            print(f"No valid data points for {base_name} vs {tgt_name}")
            continue

        fig, ax = plt.subplots(figsize=(12, 8))

        # Determine error line (bottom) and timeout line (above error) and y-limits
        correct_speedups = [s for _, s, _, _, c, t, _ in points if c and not t]
        all_speedups = [s for _, s, _, _, _, _, _ in points]
        if correct_speedups:
            min_speedup = min(correct_speedups)
        else:
            min_speedup = min(all_speedups) if all_speedups else 1.0
        max_speedup = max(all_speedups) if all_speedups else 1.0

        error_line_y = min(0.1, max(min_speedup / 5.0, 1e-12))
        timeout_line_y = error_line_y / 2.0
        y_min = timeout_line_y / 2.0
        y_max = max(max_speedup * 2.0 if max_speedup > 0 else 2.0, error_line_y * 2.0)

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
            label = f"{model}-{qtype[0]}" if label_key not in labeled else None
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
                c=color_map[model],
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

        ax.set_xscale("log")
        ax.set_xlabel("Marginal Probability")
        ax.set_ylabel(
            f"Speedup ({base_name.replace('/', '_')}/{tgt_name.replace('/', '_')})"
        )
        ax.set_yscale("log")
        ax.set_ylim(bottom=y_min, top=y_max)
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
                label=m,
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
                label=f"{qt.capitalize()}",
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
            loc="best",
            ncol=2,
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

        plt.tight_layout()
        filename = f"speedup_vs_marginal_{base_name.replace('/', '_')}_vs_{tgt_name.replace('/', '_')}.pdf"
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()
        print(f"Saved: {output_dir / filename}")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("json_file", type=Path, help="JSON file with benchmark results")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots"),
        help="Output directory for plots (default: plots/)",
    )
    args = parser.parse_args()

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

    # Generate plots
    print("\nGenerating plots...")
    plot_speedup_heatmap(results, args.output, correctness, query_type="quantitative")
    plot_speedup_heatmap(results, args.output, correctness, query_type="bounded")
    plot_speedup_vs_marginal(results, args.output, correctness)
    plot_method_scatter(results, args.output, correctness)

    print(f"\nAll plots saved to {args.output}/")


if __name__ == "__main__":
    main()
