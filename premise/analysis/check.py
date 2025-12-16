"""Trace comparison and analysis module.

This module provides tools for comparing trace files across different
configurations and models, calculating statistics, and generating plots.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Optional

# Handle both module and standalone execution
if __name__ == "__main__" and __package__ is None:
    # Running as standalone script
    _analysis_dir = Path(__file__).parent
    sys.path.insert(0, str(_analysis_dir))
    from models import MultiModelData
    from parsers import discover_folders, build_data
    from plotting import (
        plot_speed_comparison,
        plot_all_config_comparisons,
        plot_speedup_heatmap,
        plot_model_timing_boxplot,
        plot_step_runtime_scatter,
    )
    from correctness import (
        check_correctness as check_correctness_func,
        write_correctness_report,
    )
else:
    # Running as module
    from .models import MultiModelData
    from .parsers import discover_folders, build_data
    from .plotting import (
        plot_speed_comparison,
        plot_all_config_comparisons,
        plot_speedup_heatmap,
        plot_model_timing_boxplot,
        plot_step_runtime_scatter,
    )
    from .correctness import (
        check_correctness as check_correctness_func,
        generate_correctness_report,
        write_correctness_report,
    )


# =============================================================================
# Results Output Functions
# =============================================================================


def generate_summary(data: MultiModelData, path: Path) -> None:
    """Generate summary of data."""
    lines = []
    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)

    models = sorted(data.models)
    configs = sorted(data.configs)

    lines.append(f"\nModels ({len(models)}): {', '.join(models)}")
    lines.append(f"Configurations ({len(configs)}): {', '.join(configs)}")

    # ==========================================================================
    # Coverage Table: which model/config combinations exist
    # ==========================================================================
    lines.append("\n" + "=" * 80)
    lines.append("COVERAGE TABLE (✓ = found, ✗ = missing, ~ = no timing data)")
    lines.append("=" * 80)

    # Use configs as rows, models as columns (configs are longer names)
    config_width = max(len(c) for c in configs) if configs else 20

    # Header row with abbreviated model names
    header = f"{'Configuration':<{config_width}}"
    for model in models:
        # Abbreviate model names to first 10 chars
        short_model = model[:10] if len(model) > 10 else model
        header += f" {short_model:>10}"
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows (one per config)
    found_count = 0
    missing_count = 0
    no_time_count = 0
    missing_pairs = []

    for config in configs:
        row = f"{config:<{config_width}}"
        for model in models:
            if model in data.data and config in data.data[model]:
                stats = data.data[model][config]
                if stats.avg_time is not None:
                    row += f" {'✓':>10}"
                    found_count += 1
                else:
                    row += f" {'~':>10}"
                    no_time_count += 1
            else:
                row += f" {'✗':>10}"
                missing_count += 1
                missing_pairs.append((model, config))
        lines.append(row)

    total = found_count + missing_count + no_time_count
    lines.append(
        f"\nTotal: {found_count}/{total} with data ({100*found_count/total:.1f}%)"
    )
    lines.append(f"  {no_time_count} found but no timing data (~)")
    lines.append(f"  {missing_count} missing (✗)")

    # List missing combinations
    if missing_pairs:
        lines.append("\n--- Missing Combinations ---")
        for model, config in missing_pairs:
            lines.append(f"  {model} + {config}")

    # ==========================================================================
    # Average Times by Model
    # ==========================================================================
    lines.append("\n" + "=" * 80)
    lines.append("AVERAGE TIMES BY MODEL")
    lines.append("=" * 80)

    for model in models:
        if model not in data.data:
            continue
        lines.append(f"\n{model}:")
        model_configs = data.data[model]
        for config in configs:
            if config in model_configs:
                stats = model_configs[config]
                if stats.avg_time is not None:
                    lines.append(f"  {config}: {stats.avg_time:.4f}s")
                else:
                    lines.append(f"  {config}: N/A")

    summary = "\n".join(lines)
    path.write_text(summary)


# =============================================================================
# Main Comparison Functions
# =============================================================================


def compare(
    parent_folder: Path,
    folder_pattern: str = "*-unf-*",
    output_path: Optional[Path] = None,
    baseline_config: Optional[str] = None,
    check_correctness: bool = True,
) -> MultiModelData:
    """Compare statistics across models and configurations.

    This function ingests stats from models and creates combined
    comparison plots with different colors for each model.

    Args:
        parent_folder: Parent folder containing all model/config folders
        folder_pattern: Glob pattern to match folders (default: "*-unf-*")
        output_path: Where to save plots (default: parent_folder/plots)
        baseline_config: Config to use as baseline for speedup calculations
        check_correctness: Whether to check correctness of results

    Returns:
        MultiModelData structure with all parsed data
    """
    # Set up output directory (plots subdirectory)
    if output_path is None:
        output_path = parent_folder / "plots"
    output_path.mkdir(exist_ok=True)

    # Discover and parse all folders
    folders = discover_folders(parent_folder, folder_pattern)
    print(f"Found {len(folders)} folders matching pattern '{folder_pattern}'")

    if not folders:
        print("No folders found!")
        return MultiModelData()

    # Build data structure
    data = build_data(folders)

    # Print summary
    generate_summary(data, output_path / "summary.txt")

    # Check correctness if requested
    if check_correctness:
        correctness_result = check_correctness_func(data, verbose=True)
        report_path = write_correctness_report(correctness_result, output_path)
        print(f"  - Saved correctness report to {report_path.name}")

    # Generate plots
    print("\nGenerating plots...")

    # Bar chart of average times
    plot_speed_comparison(data, output_path)
    print("  - Generated speed_comparison.png")

    # Boxplot of timing distributions
    plot_model_timing_boxplot(data, output_path)
    print("  - Generated timing_boxplot.png")

    # Speedup heatmap if baseline specified
    if baseline_config:
        plot_speedup_heatmap(data, baseline_config, output_path)
        print(f"  - Generated speedup heatmap vs {baseline_config}")

    # Step runtime scatter plots for each model
    for model_name in sorted(data.models):
        plot_step_runtime_scatter(data, model_name, output_path)
    print(f"  - Generated step runtime scatter plots for {len(data.models)} models")

    # Scatter plots for all config pairs, do these last as they take longest
    plot_all_config_comparisons(data, output_path)
    print("  - Generated config comparison scatter plots")

    return data


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = ArgumentParser(
        description="Compare trace files and statistics across folders"
    )

    parser.add_argument(
        "parent_folder",
        type=Path,
        help="Parent folder containing model/config subfolders",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*-unf-*",
        help="Glob pattern to match folders (default: '*-unf-*')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder for plots (default: parent_folder/plots)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="exact-rejection-thresh=0.2",
        help="Config to use as baseline for speedup calculations",
    )
    parser.add_argument(
        "--no-correctness",
        action="store_true",
        help="Skip correctness checking (faster)",
    )

    args = parser.parse_args()

    compare(
        args.parent_folder,
        folder_pattern=args.pattern,
        output_path=args.output,
        baseline_config=args.baseline,
        check_correctness=not args.no_correctness,
    )


if __name__ == "__main__":
    main()
