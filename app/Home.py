"""CarpoolSim demo."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.jobs import summarize_saved
from app.paths import EXAMPLE_FILES
from app.paths import example_run
from app.ui import show_metrics

st.set_page_config(page_title="CarpoolSim", layout="wide")

st.title("CarpoolSim")
st.write(
    "Saved matches from the Georgia Tech 2022 off-campus commute. "
    "The demo reads those results and does not match a new day."
)

saved = example_run()
if all((saved / name).is_file() for name in EXAMPLE_FILES):
    st.subheader("Georgia Tech example")
    show_metrics(summarize_saved(saved))
    st.caption("These numbers are the saved run in data_outputs/gt_2022.")
else:
    st.info(
        "The Georgia Tech example is read from data_outputs/gt_2022. "
        "That folder needs travelers.csv, matches.csv, and legs.gpkg."
    )

st.subheader("Pages")
st.page_link("pages/1_Review.py", label="Review", help="One traveler's route.")
st.page_link("pages/2_Plots.py", label="Plots", help="Summary charts for the day.")
