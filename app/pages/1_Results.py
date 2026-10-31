"""Charts and one map for the saved Georgia Tech example."""

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
from app.paths import EXAMPLE_FILES
from app.paths import example_run
from app.ui import show_figure
from app.ui import show_metrics

st.set_page_config(page_title="Results", layout="wide")
st.title("Georgia Tech example")
st.write("Door-to-door matches from the saved Georgia Tech 2022 off-campus run.")

directory = example_run()
missing = [name for name in EXAMPLE_FILES if not (directory / name).is_file()]
if missing:
    st.info(
        "The Georgia Tech example is read from data_outputs/gt_2022. "
        "That folder needs travelers.csv, matches.csv, and legs.gpkg."
    )
    st.stop()

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
