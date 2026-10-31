"""Matched routes for the saved Georgia Tech example."""

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
from evaluation import plot_match
from evaluation import plot_vehicle_saved
from app.cache import results
from app.jobs import summarize
from app.paths import EXAMPLE_FILES
from app.paths import example_run
from app.ui import show_figure
from app.ui import show_metrics


def _match_label(row: dict[str, object], show_mode: bool) -> str:
    """Name one carpool by its travelers and the minutes saved."""
    saved = float(row["vehicle_min_saved"])
    text = (
        f"Driver {int(row['driver_id'])}, rider {int(row['rider_id'])}"
        f" · {saved:.1f} min saved"
    )
    if show_mode:
        text = f"{row['mode']} · {text}"
    return text


st.set_page_config(page_title="Results", layout="wide")
st.title("Georgia Tech example")
st.write(
    "Each carpool is the driver's pickup, the shared ride, and the dropoff. "
    "The left panel is the two drive-alone routes."
)

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

st.subheader("Matched routes")
if tables.matches.empty:
    st.info("This run has no carpools to draw.")
else:
    ordered = tables.matches.sort_values(
        ["vehicle_min_saved", "match_id"], ascending=[False, True]
    )
    show_mode = ordered["mode"].nunique(dropna=True) > 1
    labels = [_match_label(row, show_mode) for row in ordered.to_dict("records")]
    choice = st.selectbox("Carpool", labels)
    basemap = st.checkbox("Street map", value=False)
    match_id = int(ordered.iloc[labels.index(choice)]["match_id"])
    show_figure(
        plot_match(
            match_id,
            tables.travelers,
            tables.matches,
            tables.legs,
            basemap=basemap,
        )
    )

st.subheader("Travel time")
left, right = st.columns(2)
with left:
    show_figure(plot_before_after(tables.travelers))
    show_figure(plot_vehicle_saved(tables.matches))
with right:
    show_figure(plot_delta(tables.travelers))
    show_figure(plot_by_hour(tables.travelers))
