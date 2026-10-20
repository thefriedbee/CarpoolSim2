"""CarpoolSim demo."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.jobs import summarize_saved
from app.paths import root
from app.ui import show_metrics

st.set_page_config(page_title="CarpoolSim", layout="wide")

st.title("CarpoolSim")
st.write(
    "Match drive-alone commute trips into carpools. "
    "This demo uses the Georgia Tech 2022 off-campus trips on the Atlanta street network."
)

saved = root() / "data_outputs" / "gt_2022"
if (saved / "travelers.csv").is_file():
    st.subheader("Saved door-to-door run")
    show_metrics(summarize_saved(saved))
    st.caption("These numbers are already on disk in data_outputs/gt_2022.")

st.subheader("Pages")
st.page_link("pages/1_Run.py", label="Run", help="Choose a configuration and match the day.")
st.page_link("pages/2_Results.py", label="Results", help="Charts and one carpool map.")
st.page_link("pages/3_Experiments.py", label="Experiments", help="Compare the finished one-factor runs.")
