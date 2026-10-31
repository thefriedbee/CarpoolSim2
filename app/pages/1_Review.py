"""One traveler from the saved Georgia Tech example."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from evaluation import plot_trip
from app.cache import example_tables
from app.ui import show_figure


def _traveler_label(row: dict[str, object]) -> str:
    """Name one traveler by trip, role, and the change in travel time."""
    delta = float(row["delta_min"])
    return f"Trip {int(row['trip_id'])} · {str(row['role']).capitalize()} · {delta:+.1f} min"


st.set_page_config(page_title="Review", layout="wide")
st.title("Review")
st.write(
    "Pick one traveler. A matched trip shows that person's drive-alone route "
    "beside the carpool. A solo trip shows the drive alone."
)

tables = example_tables()
if tables is None:
    st.info(
        "The Georgia Tech example is read from data_outputs/gt_2022. "
        "That folder needs travelers.csv, matches.csv, and legs.gpkg."
    )
    st.stop()

if tables.travelers.empty:
    st.info("This run has no travelers to review.")
    st.stop()

ordered = tables.travelers.sort_values(["trip_id"], kind="mergesort")
labels = [_traveler_label(row) for row in ordered.to_dict("records")]
choice = st.selectbox("Traveler", labels)
person = ordered.iloc[labels.index(choice)]
role, solo, carpool, change = st.columns(4)
role.metric("Role", str(person["role"]).capitalize())
solo.metric("Solo", f"{float(person['solo_min']):.1f} min")
carpool.metric("Carpool", f"{float(person['carpool_min']):.1f} min")
change.metric("Change", f"{float(person['delta_min']):+.1f} min")

basemap = st.checkbox("Street map", value=False)
show_figure(
    plot_trip(
        int(person["trip_id"]),
        tables.travelers,
        tables.matches,
        tables.legs,
        basemap=basemap,
    )
)
