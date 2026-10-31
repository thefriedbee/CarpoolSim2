"""Keep saved tables in memory between reruns."""

from __future__ import annotations

import streamlit as st

from app.jobs import read_results


@st.cache_resource(show_spinner="Reading saved results…")
def results(directory: str, revised: float):
    """Read one run. ``revised`` is the folder time, so a replaced file reloads."""
    del revised
    return read_results(directory)
