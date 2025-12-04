"""Data models for analysis module."""

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional


@dataclass
class FolderStats:
    """Statistics parsed from a folder's stats.out file."""

    folder_path: Path
    folder_name: str
    model_name: str
    config_label: str
    states: Optional[int] = None
    transitions: Optional[int] = None
    init_time: Optional[float] = None
    promptness_deadline: Optional[int] = None
    max_time: Optional[float] = None
    min_time: Optional[float] = None
    avg_time: Optional[float] = None
    seed_times: dict[int, float] = field(default_factory=dict)
    options: Optional[str] = None

    @property
    def has_timing_stats(self) -> bool:
        """Check if timing statistics are available."""
        return self.max_time is not None and self.min_time is not None

    @property
    def is_exact(self) -> bool:
        """Check if this config uses exact arithmetic."""
        return "exact" in self.config_label

    @property
    def is_threshold(self) -> bool:
        """Check if this config uses threshold checking."""
        return "thresh" in self.config_label

    @property
    def threshold_value(self) -> Optional[float]:
        """Extract threshold value if present."""
        if "thresh=" in self.config_label:
            try:
                return float(self.config_label.split("thresh=")[1])
            except (ValueError, IndexError):
                return None
        return None


@dataclass
class TraceStats:
    """Statistics for a single folder's trace comparisons."""

    avg: list[float] = field(default_factory=list)
    avg_diff: list[float] = field(default_factory=list)
    std_diff: list[float] = field(default_factory=list)
    percent_wrong: list[float] = field(default_factory=list)
    missing: int = 0


@dataclass
class ComparisonResult:
    """Result of comparing multiple folders."""

    folder_stats: dict[str, TraceStats]
    file_stats: dict[str, FolderStats]
    worst_seed: Optional[str] = None


@dataclass
class MultiModelData:
    """Data structure for multi-model comparisons."""

    # model_name -> config_label -> FolderStats
    data: dict[str, dict[str, FolderStats]] = field(default_factory=dict)

    def add_folder_stats(self, stats: FolderStats) -> None:
        """Add folder stats organized by model and config."""
        if stats.model_name not in self.data:
            self.data[stats.model_name] = {}
        self.data[stats.model_name][stats.config_label] = stats

    @property
    def models(self) -> list[str]:
        """Get list of all model names."""
        return list(self.data.keys())

    @property
    def configs(self) -> set[str]:
        """Get set of all configuration labels across all models."""
        all_configs = set()
        for model_configs in self.data.values():
            all_configs.update(model_configs.keys())
        return all_configs

    def get_times_by_config(self, config: str) -> dict[str, dict[int, float]]:
        """Get seed times for a specific config across all models.

        Returns: {model_name: {seed: time}}
        """
        result = {}
        for model_name, configs in self.data.items():
            if config in configs and configs[config].seed_times:
                result[model_name] = configs[config].seed_times
        return result


@dataclass
class CorrectnessStats:
    """Statistics for correctness checking of a single config."""

    total_steps: int = 0
    correct_steps: int = 0
    wrong_steps: int = 0
    total_seeds: int = 0
    seeds_with_errors: int = 0
    max_error: float = 0.0
    avg_error: float = 0.0
    errors: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Percentage of correct steps."""
        if self.total_steps == 0:
            return 100.0
        return (self.correct_steps / self.total_steps) * 100

    @property
    def seed_accuracy(self) -> float:
        """Percentage of seeds with no errors."""
        if self.total_seeds == 0:
            return 100.0
        return ((self.total_seeds - self.seeds_with_errors) / self.total_seeds) * 100


@dataclass
class ModelCorrectnessResult:
    """Correctness results for a single model across all configs."""

    model_name: str
    # config_label -> CorrectnessStats
    config_stats: dict[str, CorrectnessStats] = field(default_factory=dict)
    # Reference config used for ground truth (exact method with most data)
    reference_config: Optional[str] = None
    total_seeds_checked: int = 0

    def add_config_stats(self, config: str, stats: CorrectnessStats) -> None:
        self.config_stats[config] = stats


@dataclass
class MultiModelCorrectnessResult:
    """Correctness results across all models."""

    # model_name -> ModelCorrectnessResult
    results: dict[str, ModelCorrectnessResult] = field(default_factory=dict)

    def add_model_result(self, result: ModelCorrectnessResult) -> None:
        self.results[result.model_name] = result

    @property
    def models(self) -> list[str]:
        return list(self.results.keys())

    def get_config_summary(self) -> dict[str, dict[str, float]]:
        """Get accuracy summary by config across all models.

        Returns: {config: {model: accuracy}}
        """
        summary: dict[str, dict[str, float]] = {}
        for model_name, model_result in self.results.items():
            for config, stats in model_result.config_stats.items():
                if config not in summary:
                    summary[config] = {}
                summary[config][model_name] = stats.accuracy
        return summary
