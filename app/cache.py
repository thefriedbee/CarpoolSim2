"""Keep the street network and saved tables in memory between reruns."""

from __future__ import annotations

import streamlit as st

from app.jobs import load_inputs
from app.jobs import read_results
from app.paths import root


@st.cache_resource(show_spinner="Loading the street network and trips…")
def inputs():
    """Load the Georgia Tech inputs once per server process."""
    return load_inputs(root())


@st.cache_resource(show_spinner="Reading saved results…")
def results(directory: str, revised: float):
    """Read one run. ``revised`` is the folder time, so a new run reloads."""
    del revised
    return read_results(directory)
