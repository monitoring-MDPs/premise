"""Plotting functions for analysis module."""

from itertools import combinations
from pathlib import Path
from typing import Optional

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from .models import FolderStats, MultiModelData
except ImportError:
    from models import FolderStats, MultiModelData


# Color palette for different models
MODEL_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

# Marker styles for different models
MODEL_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]


def get_model_style(model_idx: int) -> tuple[str, str]:
    """Get color and marker for a model index."""
    color = MODEL_COLORS[model_idx % len(MODEL_COLORS)]
    marker = MODEL_MARKERS[model_idx % len(MODEL_MARKERS)]
    return color, marker


def plot_speed_comparison(
    data: MultiModelData,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Plot average speed comparison across configs with different colors per model.

    Creates a grouped bar chart where each config has bars for each model.
    """
    configs = sorted(data.configs)
    models = sorted(data.models)

    if not configs or not models:
        return

    # Prepare data matrix: configs x models
    times_matrix = np.zeros((len(configs), len(models)))
    for i, config in enumerate(configs):
        for j, model in enumerate(models):
            if model in data.data and config in data.data[model]:
                stats = data.data[model][config]
                times_matrix[i, j] = stats.avg_time if stats.avg_time else 0

    # Create grouped bar chart
    x = np.arange(len(configs))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 6))

    for j, model in enumerate(models):
        color, _ = get_model_style(j)
        offset = (j - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, times_matrix[:, j], width, label=model, color=color)

    ax.set_ylabel("Average Time (s)")
    ax.set_xlabel("Configuration")
    ax.set_title(title or "Speed Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "speed_comparison.png", dpi=150)
    plt.close()


def plot_scatter(
    data: MultiModelData,
    config1: str,
    config2: str,
    output_path: Path,
) -> None:
    """Plot scatter comparison between two configs, with different colors per model."""
    models = sorted(data.models)

    fig, ax = plt.subplots(figsize=(10, 10))

    all_times = []

    for i, model in enumerate(models):
        if model not in data.data:
            continue

        model_configs = data.data[model]
        if config1 not in model_configs or config2 not in model_configs:
            continue

        stats1 = model_configs[config1]
        stats2 = model_configs[config2]

        if not stats1.seed_times or not stats2.seed_times:
            continue

        common_seeds = set(stats1.seed_times.keys()) & set(stats2.seed_times.keys())
        if not common_seeds:
            continue

        x = [stats1.seed_times[seed] for seed in common_seeds]
        y = [stats2.seed_times[seed] for seed in common_seeds]

        all_times.extend(x)
        all_times.extend(y)

        color, marker = get_model_style(i)
        ax.scatter(x, y, c=color, marker=marker, label=model, alpha=0.7, s=50)

    if not all_times:
        plt.close()
        return

    max_time = max(all_times)
    min_time = min(all_times)

    # Reference lines
    ax.plot([min_time, max_time], [min_time, max_time], "k-", linewidth=1, label="1:1")
    ax.plot(
        [min_time, max_time],
        [min_time * 10, max_time * 10],
        "k--",
        linewidth=0.5,
        label="10x",
    )
    ax.plot(
        [min_time, max_time], [min_time * 0.1, max_time * 0.1], "k--", linewidth=0.5
    )
    ax.plot(
        [min_time, max_time],
        [min_time * 100, max_time * 100],
        "k:",
        linewidth=0.5,
        label="100x",
    )
    ax.plot(
        [min_time, max_time], [min_time * 0.01, max_time * 0.01], "k:", linewidth=0.5
    )

    ax.set_xlim(min_time * 0.8, max_time * 1.2)
    ax.set_ylim(min_time * 0.8, max_time * 1.2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Time for {config1} (s)")
    ax.set_ylabel(f"Time for {config2} (s)")
    ax.set_title(
        f"Seed Time Comparison: {config1} vs {config2}\n(colors = different models)"
    )
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()

    safe_config1 = config1.replace("/", "-").replace("=", "-")
    safe_config2 = config2.replace("/", "-").replace("=", "-")
    plt.savefig(
        output_path / f"scatter_{safe_config1}_vs_{safe_config2}.png",
        dpi=150,
    )
    plt.close()


def plot_all_config_comparisons(
    data: MultiModelData,
    output_path: Path,
) -> None:
    """Generate scatter plots for all pairs of configurations."""
    configs = sorted(data.configs)

    for config1, config2 in combinations(configs, 2):
        plot_scatter(data, config1, config2, output_path)


def plot_speedup_heatmap(
    multi_data: MultiModelData,
    baseline_config: str,
    output_path: Path,
) -> None:
    """Plot heatmap showing speedup of each config relative to baseline."""
    configs = sorted(multi_data.configs)
    models = sorted(multi_data.models)

    if baseline_config not in configs:
        print(f"Baseline config '{baseline_config}' not found")
        return

    other_configs = [c for c in configs if c != baseline_config]

    # Calculate speedups: baseline_time / config_time
    speedup_matrix = np.zeros((len(other_configs), len(models)))

    for i, config in enumerate(other_configs):
        for j, model in enumerate(models):
            if model not in multi_data.data:
                continue

            model_configs = multi_data.data[model]
            if baseline_config not in model_configs or config not in model_configs:
                continue

            baseline_stats = model_configs[baseline_config]
            config_stats = model_configs[config]

            if baseline_stats.avg_time and config_stats.avg_time:
                speedup_matrix[i, j] = baseline_stats.avg_time / config_stats.avg_time

    speedup_matrix = np.ma.masked_equal(speedup_matrix, 0)  # Mask zeros

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(speedup_matrix, cmap="RdYlGn", aspect="auto", norm=LogNorm(vmin=0.1))

    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(other_configs)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticklabels(other_configs)

    # Add text annotations
    for i in range(len(other_configs)):
        for j in range(len(models)):
            if speedup_matrix.mask[i, j]:
                ax.text(
                    j, i, "N/A", ha="center", va="center", color="black", fontsize=8
                )
            else:
                text = f"{speedup_matrix[i, j]:.2f}x"
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Configuration")
    ax.set_title(f"Speedup relative to {baseline_config}")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Speedup (baseline_time / config_time)")

    plt.tight_layout()
    plt.savefig(
        output_path
        / f"speedup_heatmap_vs_{baseline_config.replace('/', '-').replace('=', '-')}.png",
        dpi=150,
    )
    plt.close()


def plot_model_timing_boxplot(
    data: MultiModelData,
    output_path: Path,
) -> None:
    """Plot boxplot of timing distributions with models on x-axis, colored by config."""
    configs = sorted(data.configs)
    models = sorted(data.models)

    fig, ax = plt.subplots(figsize=(14, 6))

    positions = []
    box_data = []
    colors = []

    pos = 0
    group_positions = []

    for model in models:
        group_start = pos
        for j, config in enumerate(configs):
            if model in data.data and config in data.data[model]:
                stats = data.data[model][config]
                if stats.seed_times:
                    times = list(stats.seed_times.values())
                    box_data.append(times)
                    positions.append(pos)
                    color, _ = get_model_style(j)
                    colors.append(color)
                    pos += 1

        if pos > group_start:
            group_positions.append((group_start + pos - 1) / 2)
        pos += 0.5  # Gap between groups

    if not box_data:
        plt.close()
        return

    bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.7)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add vertical lines between model groups
    for i in range(len(group_positions) - 1):
        # Find the midpoint between this group and the next
        if i + 1 < len(group_positions):
            line_pos = (group_positions[i] + group_positions[i + 1]) / 2
            ax.axvline(
                x=line_pos, color="gray", linestyle="--", linewidth=0.8, alpha=0.5
            )

    ax.set_xticks(group_positions)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Time (s)")
    ax.set_yscale("log")
    ax.set_title("Timing Distribution by Model and Configuration")

    # Create legend for configs
    legend_handles = []
    for j, config in enumerate(configs):
        color, _ = get_model_style(j)
        legend_handles.append(
            mpatches.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7, label=config)
        )
    ax.legend(
        handles=legend_handles,
        title="Configuration",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.savefig(output_path / "timing_boxplot.png", dpi=150)
    plt.close()


def plot_step_runtime_scatter(
    data: MultiModelData,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot step runtime for a single model, with configs as different colors.

    X-axis: Step index
    Y-axis: Time per step (log scale)
    Colors: Different configs
    Each seed is plotted as a connected line with scatter points.
    """
    if model_name not in data.data:
        print(f"Model '{model_name}' not found")
        return

    model_configs = data.data[model_name]
    configs = sorted(model_configs.keys())

    if not configs:
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    has_data = False

    for j, config in enumerate(configs):
        stats = model_configs[config]
        if not stats.time_per_step:
            continue

        color, marker = get_model_style(j)
        first_seed = True

        # Plot each seed as a connected line
        for seed, step_times in stats.time_per_step.items():
            if not step_times:
                continue

            has_data = True
            step_indices = list(range(len(step_times)))

            # Plot line connecting points of same seed
            ax.plot(
                step_indices,
                step_times,
                color=color,
                alpha=0.3,
                linewidth=0.8,
            )
            # Plot scatter points
            ax.scatter(
                step_indices,
                step_times,
                c=color,
                marker=marker,
                label=config if first_seed else None,
                alpha=0.5,
                s=10,
            )
            first_seed = False

    if not has_data:
        plt.close()
        return

    ax.set_xlabel("Step Index")
    ax.set_ylabel("Time per Step (s)")
    ax.set_yscale("log")
    ax.set_title(f"Step Runtime by Index - {model_name}")
    ax.legend(title="Configuration", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_model = model_name.replace("/", "-").replace("=", "-")
    plt.savefig(output_path / f"step_runtime_{safe_model}.png", dpi=150)
    plt.close()
