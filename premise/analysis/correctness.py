"""Correctness checking for trace comparisons across methods."""

import math
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Optional

from tqdm import tqdm

try:
    from .models import (
        FolderStats,
        MultiModelData,
        CorrectnessStats,
        ModelCorrectnessResult,
        MultiModelCorrectnessResult,
    )
    from .parsers import parse_trace_file
except ImportError:
    from models import (
        FolderStats,
        MultiModelData,
        CorrectnessStats,
        ModelCorrectnessResult,
        MultiModelCorrectnessResult,
    )
    from parsers import parse_trace_file


# Tolerance for float comparisons
FLOAT_TOLERANCE = 1e-4


def extract_seed_from_filename(filename: str) -> Optional[str]:
    """Extract seed from trace filename."""
    match = re.search(r"-(\d+)\.csv$", filename)
    return match.group(1) if match else None


def discover_traces_by_model_and_seed(
    multi_data: MultiModelData,
) -> dict[str, dict[str, dict[str, Path]]]:
    """Discover all trace files organized by model, seed, and config.

    Returns: {model_name: {seed: {config_label: trace_path}}}
    """
    result: dict[str, dict[str, dict[str, Path]]] = {}

    for model_name, configs in multi_data.data.items():
        result[model_name] = {}

        for config_label, folder_stats in configs.items():
            folder_path = folder_stats.folder_path

            for trace_file in folder_path.glob("trace-*.csv"):
                seed = extract_seed_from_filename(trace_file.name)
                if seed:
                    if seed not in result[model_name]:
                        result[model_name][seed] = {}
                    result[model_name][seed][config_label] = trace_file

    return result


def get_reference_config(configs: dict[str, FolderStats]) -> Optional[str]:
    """Select the best reference config for ground truth.

    Prioritizes exact methods without threshold (gives actual values).
    """
    # Priority: exact without threshold > exact with threshold
    exact_no_thresh = [c for c in configs if "exact" in c and "thresh" not in c]
    if exact_no_thresh:
        # Prefer bisection > restart > rejection for stability
        for method in ["bisection", "restart", "rejection"]:
            for c in exact_no_thresh:
                if method in c:
                    return c
        return exact_no_thresh[0]

    # Fall back to any exact method (even with threshold, we can derive values)
    # This is less ideal but better than nothing
    exact_any = [c for c in configs if "exact" in c]
    if exact_any:
        return exact_any[0]

    return None


def compute_ground_truth_risks(
    traces_by_config: dict[str, list[dict]],
    configs: dict[str, FolderStats],
) -> tuple[list[Fraction], Optional[float]]:
    """Compute ground truth risk values using majority voting on exact methods.

    Returns:
        Tuple of (ground_truth_risks, threshold_value)
        - ground_truth_risks: List of Fraction values for each step
        - threshold_value: The threshold if threshold configs are present
    """
    # Separate exact (non-threshold) traces for ground truth
    exact_traces = []
    threshold_value = None

    for config, trace in traces_by_config.items():
        config_stats = configs.get(config)
        if config_stats and config_stats.is_exact and not config_stats.is_threshold:
            exact_traces.append((config, trace))
        if config_stats and config_stats.threshold_value is not None:
            threshold_value = config_stats.threshold_value

    if not exact_traces:
        return [], threshold_value

    # Use the first exact trace as reference length
    ref_trace = exact_traces[0][1]
    ground_truth = []

    for step_idx in range(len(ref_trace)):
        step_risks = []
        for config, trace in exact_traces:
            if step_idx < len(trace):
                risk = trace[step_idx]["risk"]
                if isinstance(risk, Fraction):
                    step_risks.append(risk)

        if step_risks:
            # Majority voting
            risk_counts: dict[Fraction, int] = {}
            for r in step_risks:
                risk_counts[r] = risk_counts.get(r, 0) + 1
            best_risk = max(risk_counts.items(), key=lambda x: x[1])[0]
            ground_truth.append(best_risk)
        else:
            # No exact values available, use 0 as placeholder
            ground_truth.append(Fraction(0))

    return ground_truth, threshold_value


def check_risk_correctness(
    computed_risk: Fraction | float | bool,
    ground_truth: Fraction,
    threshold: Optional[float] = None,
    is_threshold_config: bool = False,
) -> tuple[bool, float]:
    """Check if a computed risk is correct against ground truth.

    Args:
        computed_risk: The risk value from the method being checked
        ground_truth: The ground truth Fraction value
        threshold: The threshold value if applicable
        is_threshold_config: Whether this config uses threshold checking

    Returns:
        Tuple of (is_correct, error_magnitude)
    """
    if isinstance(computed_risk, bool):
        # Threshold result - check if it matches expected threshold comparison
        # Semantics: True = risk < threshold (safe), False = risk >= threshold (dangerous)
        if threshold is not None:
            expected = float(ground_truth) < threshold
            is_correct = computed_risk == expected
            error = 0.0 if is_correct else 1.0
            return is_correct, error
        else:
            # No threshold specified, can't verify
            return True, 0.0

    elif isinstance(computed_risk, Fraction):
        # Exact value - should match exactly
        error = abs(float(computed_risk - ground_truth))
        is_correct = computed_risk == ground_truth
        return is_correct, error

    elif isinstance(computed_risk, float):
        # Float value - check within tolerance
        gt_float = float(ground_truth)
        error = abs(computed_risk - gt_float)
        is_correct = math.isclose(computed_risk, gt_float, abs_tol=FLOAT_TOLERANCE)
        return is_correct, error

    return True, 0.0


def check_model_correctness(
    model_name: str,
    seeds_by_config: dict[str, dict[str, Path]],
    configs: dict[str, FolderStats],
) -> ModelCorrectnessResult:
    """Check correctness for a single model across all seeds and configs.

    Args:
        model_name: Name of the model
        seeds_by_config: {seed: {config: trace_path}}
        configs: {config_label: FolderStats}

    Returns:
        ModelCorrectnessResult with stats for each config
    """
    result = ModelCorrectnessResult(model_name=model_name)
    result.reference_config = get_reference_config(configs)

    # Initialize stats for each config
    config_stats: dict[str, CorrectnessStats] = {
        config: CorrectnessStats() for config in configs
    }

    seeds_checked = 0

    for seed, config_paths in seeds_by_config.items():
        # Load all traces for this seed
        traces_by_config: dict[str, list[dict]] = {}
        for config, path in config_paths.items():
            try:
                traces_by_config[config] = parse_trace_file(path)
            except Exception as e:
                print(f"Warning: Failed to parse {path}: {e}")
                continue

        if not traces_by_config:
            continue

        # Compute ground truth from exact methods (non-threshold)
        ground_truth, _ = compute_ground_truth_risks(traces_by_config, configs)

        if not ground_truth:
            continue

        seeds_checked += 1

        # Check each config against ground truth
        for config, trace in traces_by_config.items():
            stats = config_stats[config]
            stats.total_seeds += 1

            config_info = configs.get(config)
            is_threshold = config_info.is_threshold if config_info else False
            threshold_value = config_info.threshold_value if config_info else None

            seed_has_error = False

            for step_idx, step in enumerate(trace):
                if step_idx >= len(ground_truth):
                    break

                gt = ground_truth[step_idx]
                computed = step["risk"]

                is_correct, error = check_risk_correctness(
                    computed, gt, threshold_value, is_threshold
                )

                stats.total_steps += 1

                if is_correct:
                    stats.correct_steps += 1
                else:
                    stats.wrong_steps += 1
                    stats.errors.append(error)
                    seed_has_error = True
                    if error > stats.max_error:
                        stats.max_error = error

            if seed_has_error:
                stats.seeds_with_errors += 1

    # Compute average errors
    for config, stats in config_stats.items():
        if stats.errors:
            stats.avg_error = sum(stats.errors) / len(stats.errors)
        result.add_config_stats(config, stats)

    result.total_seeds_checked = seeds_checked
    return result


def check_correctness(
    data: MultiModelData,
    verbose: bool = True,
) -> MultiModelCorrectnessResult:
    """Check correctness across all models and configurations.

    Args:
        data: The data structure
        verbose: Whether to print progress

    Returns:
        MultiModelCorrectnessResult with all correctness statistics
    """
    result = MultiModelCorrectnessResult()

    # Discover all traces organized by model and seed
    if verbose:
        print("\nDiscovering trace files...")
    traces_by_model = discover_traces_by_model_and_seed(data)

    # Check each model
    models = list(traces_by_model.keys())
    iterator = tqdm(models, desc="Checking correctness") if verbose else models

    for model_name in iterator:
        seeds_by_config = traces_by_model[model_name]
        configs = data.data.get(model_name, {})

        if not configs:
            continue

        model_result = check_model_correctness(model_name, seeds_by_config, configs)
        result.add_model_result(model_result)

    return result


def generate_correctness_report(result: MultiModelCorrectnessResult) -> str:
    """Generate correctness report as a string.

    Structure:
    1. Summary table (configs x models)
    2. Aggregate stats per config (all models combined)
    3. Per-model detailed breakdown
    """
    lines = []

    # ==========================================================================
    # SECTION 1: SUMMARY TABLE
    # ==========================================================================
    lines.append("=" * 70)
    lines.append("CORRECTNESS SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    summary = result.get_config_summary()
    configs = sorted(summary.keys())
    models = sorted(result.models)

    # Header
    header = f"{'Config':<40}"
    for model in models:
        short_model = model[:12] if len(model) > 12 else model
        header += f" {short_model:>12}"
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for config in configs:
        row = f"{config:<40}"
        for model in models:
            if model in summary[config]:
                acc = summary[config][model]
                marker = "✓" if acc == 100.0 else "✗"
                row += f" {marker}{acc:>10.1f}%"
            else:
                row += f" {'--':>12}"
        lines.append(row)

    # ==========================================================================
    # SECTION 2: AGGREGATE STATS PER CONFIG (ALL MODELS COMBINED)
    # ==========================================================================
    lines.append("")
    lines.append("=" * 70)
    lines.append("AGGREGATE STATS BY CONFIGURATION (ALL MODELS)")
    lines.append("=" * 70)
    lines.append("")

    # Aggregate stats across all models for each config
    config_aggregates: dict[str, dict] = {}

    for config in configs:
        agg = {
            "total_steps": 0,
            "correct_steps": 0,
            "total_seeds": 0,
            "seeds_with_errors": 0,
            "all_errors": [],
            "max_error": 0.0,
        }

        for model_name, model_result in result.results.items():
            if config in model_result.config_stats:
                stats = model_result.config_stats[config]
                agg["total_steps"] += stats.total_steps
                agg["correct_steps"] += stats.correct_steps
                agg["total_seeds"] += stats.total_seeds
                agg["seeds_with_errors"] += stats.seeds_with_errors
                agg["all_errors"].extend(stats.errors)
                if stats.max_error > agg["max_error"]:
                    agg["max_error"] = stats.max_error

        if agg["total_steps"] > 0:
            agg["accuracy"] = (agg["correct_steps"] / agg["total_steps"]) * 100
        else:
            agg["accuracy"] = 0.0

        if agg["all_errors"]:
            agg["avg_error"] = sum(agg["all_errors"]) / len(agg["all_errors"])
        else:
            agg["avg_error"] = 0.0

        config_aggregates[config] = agg

    # Print aggregate stats
    for config in configs:
        agg = config_aggregates[config]
        if agg["total_steps"] == 0:
            lines.append(f"{config}: No data")
            continue

        status = "✓" if agg["accuracy"] == 100.0 else "✗"
        is_threshold = "thresh" in config
        lines.append(f"{status} {config}:")
        lines.append(
            f"    Steps: {agg['correct_steps']}/{agg['total_steps']} correct ({agg['accuracy']:.2f}%)"
        )
        lines.append(
            f"    Seeds: {agg['total_seeds'] - agg['seeds_with_errors']}/{agg['total_seeds']} perfect"
        )

        if agg["all_errors"]:
            if is_threshold:
                wrong_pct = 100.0 - agg["accuracy"]
                lines.append(f"    Wrong: {wrong_pct:.2f}%")
            else:
                lines.append(f"    Max error: {agg['max_error']:.2e}")
                lines.append(f"    Avg error: {agg['avg_error']:.2e}")
        lines.append("")

    # ==========================================================================
    # SECTION 3: PER-MODEL DETAILED BREAKDOWN
    # ==========================================================================
    lines.append("=" * 70)
    lines.append("DETAILED STATS BY MODEL")
    lines.append("=" * 70)

    for model_name, model_result in sorted(result.results.items()):
        lines.append("")
        lines.append("─" * 70)
        lines.append(f"Model: {model_name}")
        lines.append(f"Reference config: {model_result.reference_config}")
        lines.append(f"Seeds checked: {model_result.total_seeds_checked}")
        lines.append("─" * 70)

        sorted_configs = sorted(model_result.config_stats.items())

        for config, stats in sorted_configs:
            if stats.total_steps == 0:
                lines.append(f"  {config}: No data")
                continue

            status = "✓" if stats.accuracy == 100.0 else "✗"
            is_threshold = "thresh" in config
            lines.append(f"  {status} {config}:")
            lines.append(
                f"      Steps: {stats.correct_steps}/{stats.total_steps} correct ({stats.accuracy:.2f}%)"
            )
            lines.append(
                f"      Seeds: {stats.total_seeds - stats.seeds_with_errors}/{stats.total_seeds} perfect ({stats.seed_accuracy:.2f}%)"
            )

            if stats.wrong_steps > 0:
                if is_threshold:
                    wrong_pct = 100.0 - stats.accuracy
                    lines.append(f"      Wrong: {wrong_pct:.2f}%")
                else:
                    lines.append(f"      Max error: {stats.max_error:.2e}")
                    lines.append(f"      Avg error: {stats.avg_error:.2e}")

    return "\n".join(lines)


def write_correctness_report(
    result: MultiModelCorrectnessResult,
    output_path: Path,
    filename: str = "correctness_report.txt",
) -> Path:
    """Write correctness report to a text file.

    Args:
        result: The correctness results
        output_path: Directory to write the report to
        filename: Name of the report file

    Returns:
        Path to the written report file
    """
    report = generate_correctness_report(result)
    report_path = output_path / filename
    report_path.write_text(report)
    return report_path
