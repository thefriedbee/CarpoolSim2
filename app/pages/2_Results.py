"""Charts and one map for a finished run."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from evaluation import plot_before_after
from evaluation import plot_by_hour
from evaluation import plot_day
from evaluation import plot_delta
from evaluation import plot_match
from evaluation import plot_vehicle_saved
from app.cache import results
from app.jobs import summarize
from app.paths import root
from app.paths import saved_runs
from app.ui import show_figure
from app.ui import show_metrics

st.set_page_config(page_title="Results", layout="wide")
st.title("Results")

runs = saved_runs()
if not runs:
    st.info("No saved runs yet. Use Run to match the day.")
    st.stop()

project = root()
labels = [str(path.relative_to(project)) for path in runs]
default = labels[0]
for label in labels:
    if label == "data_outputs/gt_2022":
        default = label
        break

fresh = st.session_state.pop("result_dir", None)
if fresh:
    fresh_label = str(Path(str(fresh)).resolve().relative_to(project))
    if fresh_label in labels:
        st.session_state["saved_run"] = fresh_label
elif "saved_run" not in st.session_state or st.session_state["saved_run"] not in labels:
    st.session_state["saved_run"] = default

choice = st.selectbox("Saved run", labels, key="saved_run")
directory = runs[labels.index(choice)]
revised = (directory / "travelers.csv").stat().st_mtime
tables = results(str(directory), revised)
show_metrics(summarize(tables.travelers, tables.matches))

charts, maps = st.tabs(["Charts", "Maps"])

with charts:
    left, right = st.columns(2)
    with left:
        show_figure(plot_before_after(tables.travelers))
        show_figure(plot_vehicle_saved(tables.matches))
    with right:
        show_figure(plot_delta(tables.travelers))
        show_figure(plot_by_hour(tables.travelers))

with maps:
    basemap = st.checkbox("Background map", value=False)
    show_figure(plot_day(tables.travelers, tables.matches, basemap=basemap))
    if tables.matches.empty:
        st.info("This run has no carpools to draw.")
    else:
        match_ids = [int(match_id) for match_id in tables.matches["match_id"]]
        match_id = st.selectbox("Carpool", match_ids)
        show_figure(
            plot_match(
                match_id,
                tables.travelers,
                tables.matches,
                tables.legs,
                basemap=basemap,
            )
        )
