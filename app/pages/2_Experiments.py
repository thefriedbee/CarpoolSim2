"""Compare the finished one-factor experiments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.jobs import experiment_table
from app.paths import root

st.set_page_config(page_title="Experiments", layout="wide")
st.title("Experiments")
st.write(
    "Each row changes one setting from that mode's baseline. "
    "The table reads the finished summary and does not rerun the notebooks."
)

table = experiment_table(root())
if table is None or table.empty:
    st.info("No experiment summary was found at data_outputs/experiments/summary.csv.")
    st.stop()

table = table.copy()
if "match_rate" in table:
    table["match_rate"] = table["match_rate"].map(lambda value: "" if pd.isna(value) else f"{value:.1%}")
if "vehicle_min_saved" in table:
    table["vehicle_min_saved"] = table["vehicle_min_saved"].round(0)
if "simulation_seconds" in table:
    table["simulation_seconds"] = table["simulation_seconds"].round(1)

st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "experiment_id": "Experiment",
        "mode": "Mode",
        "parameter": "Parameter",
        "baseline_value": "Baseline",
        "value": "Value",
        "match_rate": "Match rate",
        "vehicle_min_saved": "Minutes saved",
        "simulation_seconds": "Seconds",
        "description": "Description",
    },
)
