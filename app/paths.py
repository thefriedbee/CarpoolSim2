"""Project paths for the demo. This module does not import Streamlit."""

from __future__ import annotations

from pathlib import Path

EXAMPLE_FILES = ("travelers.csv", "matches.csv", "legs.gpkg")


def root() -> Path:
    """Return the repository root that contains ``pyproject.toml``."""
    return Path(__file__).resolve().parents[1]


def example_run() -> Path:
    """Saved Georgia Tech results the demo displays.

    The folder holds ``travelers.csv``, ``matches.csv``, and ``legs.gpkg``.
    """
    return root() / "data_outputs" / "gt_2022"


def saved_runs() -> list[Path]:
    """Return output folders that contain the three result tables."""
    outputs = root() / "data_outputs"
    if not outputs.is_dir():
        return []
    runs = []
    for travelers in sorted(outputs.rglob("travelers.csv")):
        folder = travelers.parent
        if (folder / "matches.csv").is_file() and (folder / "legs.gpkg").is_file():
            runs.append(folder)
    return runs
