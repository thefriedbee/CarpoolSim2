"""Read saved carpool results.

This module does not import Streamlit. The pages call it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.score import RunTables
from evaluation.score import load_results


def read_results(directory: str | Path) -> RunTables:
    """Read travelers, carpools, and leg lines from one output folder."""
    return load_results(directory)


def summarize(travelers: pd.DataFrame, matches: pd.DataFrame) -> dict[str, object]:
    """Count travelers, carpools, the match rate, and minutes of driving saved."""
    count = int(len(travelers))
    solo = int((travelers["role"] == "solo").sum()) if count else 0
    matched = count - solo
    saved = 0.0 if matches.empty else float(matches["vehicle_min_saved"].sum())
    return {
        "travelers": count,
        "carpools": int(len(matches)),
        "solo": solo,
        "matched": matched,
        "match_rate": (matched / count) if count else 0.0,
        "minutes_saved": saved,
    }


def summarize_saved(directory: Path) -> dict[str, object]:
    """Summarize a finished run from its CSV tables."""
    travelers = pd.read_csv(directory / "travelers.csv")
    matches = pd.read_csv(directory / "matches.csv")
    return summarize(travelers, matches)


def experiment_table(project: Path) -> pd.DataFrame | None:
    """Return the finished one-factor runs, or ``None`` when the file is absent."""
    path = project / "data_outputs" / "experiments" / "summary.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    columns = [
        "experiment_id",
        "mode",
        "parameter",
        "baseline_value",
        "value",
        "match_rate",
        "vehicle_min_saved",
        "simulation_seconds",
        "description",
    ]
    return frame[[column for column in columns if column in frame.columns]]
