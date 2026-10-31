"""Keep saved tables in memory between reruns."""

from __future__ import annotations

import streamlit as st

from app.jobs import read_results
from app.paths import EXAMPLE_FILES
from app.paths import example_run


@st.cache_resource(show_spinner="Reading saved results…")
def results(directory: str, revised: float):
    """Read one run. ``revised`` is the folder time, so a replaced file reloads."""
    del revised
    return read_results(directory)


def example_tables():
    """Read the Georgia Tech example, or ``None`` when a table is missing."""
    directory = example_run()
    if not all((directory / name).is_file() for name in EXAMPLE_FILES):
        return None
    revised = (directory / "travelers.csv").stat().st_mtime
    return results(str(directory), revised)
