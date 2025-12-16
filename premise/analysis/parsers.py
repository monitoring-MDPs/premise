"""File and folder parsing utilities for analysis module."""

import re
from csv import DictReader
from fractions import Fraction
from pathlib import Path
from typing import Optional

try:
    from .models import FolderStats, MultiModelData
except ImportError:
    from models import FolderStats, MultiModelData


def parse_folder_name(folder_name: str) -> tuple[str, str]:
    """Parse folder name into model name and config label.

    Handles patterns like:
    - 'airportA-7-400-40-unf-float-bisection'
    - 'airportA-7-400-40-unf-float-bisection-thresh=0.2'
    - 'evadeI-10-unf-exact-rejection'

    Returns: (model_name, config_label)
    """
    # Try to find '-unf-' which separates model from config
    if "-unf-" in folder_name:
        parts = folder_name.split("-unf-", 1)
        return parts[0], parts[1]

    # Fallback: try other common separators
    for sep in ["-ff-", "-exact-", "-float-"]:
        if sep in folder_name:
            parts = folder_name.split(sep, 1)
            return parts[0], sep.strip("-") + "-" + parts[1]

    # Last resort: return the whole name as model
    return folder_name, "unknown"


def parse_stats_file(stats_path: Path) -> dict:
    """Parse a stats.out file and return extracted values."""
    result = {
        "states": None,
        "transitions": None,
        "init_time": None,
        "promptness_deadline": None,
        "max_time": None,
        "min_time": None,
        "avg_time": None,
        "seed_times": {},
        "time_per_step": {},
        "options": None,
    }

    if not stats_path.exists():
        return result

    with open(stats_path, "r") as f:
        next_line_all_times = False
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if key == "states":
                    result["states"] = int(value)
                elif key == "transitions":
                    result["transitions"] = int(value)
                elif key == "init_time":
                    result["init_time"] = float(value)
                elif key == "promptness_deadline":
                    result["promptness_deadline"] = int(value)
                elif key == "max_time":
                    result["max_time"] = float(value)
                elif key == "min_time":
                    result["min_time"] = float(value)
                elif key == "avg_time":
                    result["avg_time"] = float(value)
                elif key == "options":
                    result["options"] = value
                elif key == "all_times":
                    for pair in value.strip()[1:-1].split(","):
                        seed_str, time_str = pair.strip().split(":", 1)
                        result["seed_times"][int(seed_str.strip())] = float(
                            time_str.strip()
                        )
                elif key == "time_per_step":
                    result["time_per_step"] = eval(value)

    return result


def parse_folder(folder: Path) -> FolderStats:
    """Parse a folder and return its statistics."""
    folder_name = folder.name
    model_name, config_label = parse_folder_name(folder_name)

    stats_path = folder / "stats.out"
    parsed = parse_stats_file(stats_path)

    return FolderStats(
        folder_path=folder,
        folder_name=folder_name,
        model_name=model_name,
        config_label=config_label,
        states=parsed["states"],
        transitions=parsed["transitions"],
        init_time=parsed["init_time"],
        promptness_deadline=parsed["promptness_deadline"],
        max_time=parsed["max_time"],
        min_time=parsed["min_time"],
        avg_time=parsed["avg_time"],
        seed_times=parsed["seed_times"],
        time_per_step=parsed["time_per_step"],
        options=parsed["options"],
    )


def discover_folders(parent: Path, pattern: str = "*-unf-*") -> list[Path]:
    """Discover all matching folders in a parent directory."""
    return [f for f in parent.glob(pattern) if f.is_dir()]


def parse_all_folders(folders: list[Path]) -> dict[str, FolderStats]:
    """Parse all folders and return a dict of folder_name -> FolderStats."""
    return {folder.name: parse_folder(folder) for folder in folders}


def build_data(folders: list[Path]) -> MultiModelData:
    """Build a MultiModelData structure from a list of folders."""
    data = MultiModelData()
    for folder in folders:
        stats = parse_folder(folder)
        data.add_folder_stats(stats)
    return data


def discover_trace_files(folders: list[Path]) -> dict[str, dict[str, Path]]:
    """Discover all trace CSV files across folders.

    Returns: {seed: {folder_name: path}}
    """
    seed_regex = r".*-(\d+)\.csv$"
    paths: dict[str, dict[str, Path]] = {}

    for folder in folders:
        folder_name = folder.name
        for path in folder.glob("*.csv"):
            match = re.match(seed_regex, path.name)
            if match:
                seed = match.group(1)
                if seed not in paths:
                    paths[seed] = {}
                paths[seed][folder_name] = path

    return paths


def parse_trace_file(path: Path) -> list[dict]:
    """Parse a trace CSV file and return list of steps with parsed risk values."""
    with open(path, "r", newline="") as f:
        trace_reader = DictReader(f)
        trace = list(trace_reader)

    # Parse risk values
    for step in trace:
        risk_str = step["risk"]
        if "." in risk_str:
            step["risk"] = float(risk_str)
        elif risk_str in ("True", "False"):
            step["risk"] = risk_str == "True"
        else:
            step["risk"] = Fraction(risk_str)

    return trace
