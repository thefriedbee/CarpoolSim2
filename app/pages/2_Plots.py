"""Summary charts for the saved Georgia Tech example."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from evaluation import plot_before_after
from evaluation import plot_by_hour
from evaluation import plot_delta
from evaluation import plot_vehicle_saved
from app.cache import example_tables
from app.jobs import summarize
from app.ui import show_figure
from app.ui import show_metrics

st.set_page_config(page_title="Plots", layout="wide")
st.title("Plots")
st.write(
    "Travel time before and after carpooling, the change for drivers and riders, "
    "minutes saved by each carpool, and matches by departure hour."
)

tables = example_tables()
if tables is None:
    st.info(
        "The Georgia Tech example is read from data_outputs/gt_2022. "
        "That folder needs travelers.csv, matches.csv, and legs.gpkg."
    )
    st.stop()

show_metrics(summarize(tables.travelers, tables.matches))

left, right = st.columns(2)
with left:
    show_figure(plot_before_after(tables.travelers))
    show_figure(plot_vehicle_saved(tables.matches))
with right:
    show_figure(plot_delta(tables.travelers))
    show_figure(plot_by_hour(tables.travelers))
