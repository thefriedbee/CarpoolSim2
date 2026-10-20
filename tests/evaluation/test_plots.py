"""Tests for the before/after charts and maps."""

import matplotlib

matplotlib.use("Agg", force=True)

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from shapely.geometry import LineString

from evaluation.maps import plot_day
from evaluation.maps import plot_match
from evaluation.plots import plot_before_after
from evaluation.plots import plot_by_hour
from evaluation.plots import plot_delta
from evaluation.plots import plot_vehicle_saved


@pytest.fixture(autouse=True)
def close_figures():
    """Drop figures so one test does not keep the next test's canvas."""
    yield
    plt.close("all")


def travelers() -> pd.DataFrame:
    """Three people in one direct carpool morning, plus a later solo trip."""
    return pd.DataFrame(
        {
            "trip_id": [1, 2, 3],
            "role": ["driver", "rider", "solo"],
            "mode": ["direct", "direct", ""],
            "departure_min": [7 * 60 + 30, 7 * 60 + 40, 9 * 60],
            "solo_min": [20.0, 15.0, 30.0],
            "carpool_min": [24.0, 18.0, 30.0],
            "delta_min": [4.0, 3.0, 0.0],
            "origin_x": [0.0, 1000.0, 500.0],
            "origin_y": [0.0, 0.0, 1000.0],
            "dest_x": [4000.0, 3000.0, 1500.0],
            "dest_y": [0.0, 0.0, 1000.0],
        }
    )


def matches() -> pd.DataFrame:
    """The carpool that pairs traveler 1 with traveler 2."""
    return pd.DataFrame(
        {
            "match_id": [0],
            "driver_id": [1],
            "rider_id": [2],
            "mode": ["direct"],
            "lot_x": [float("nan")],
            "lot_y": [float("nan")],
            "vehicle_min_saved": [8.0],
        }
    )


def legs() -> gpd.GeoDataFrame:
    """Solo paths and a pickup that ends where the shared ride starts."""
    return gpd.GeoDataFrame(
        {
            "match_id": [0, 0, 0, 0, 0],
            "who": ["driver", "rider", "driver", "driver", "driver"],
            "leg": ["solo", "solo", "pickup", "shared", "dropoff"],
            "travel_min": [4.0, 2.0, 1.0, 2.0, 1.0],
        },
        geometry=[
            LineString([(0, 0), (4000, 0)]),
            LineString([(1000, 0), (3000, 0)]),
            LineString([(0, 0), (1000, 0)]),
            LineString([(1000, 0), (3000, 0)]),
            LineString([(3000, 0), (4000, 0)]),
        ],
        crs="EPSG:3857",
    )


def test_before_after_draws_the_identity_line() -> None:
    fig = plot_before_after(travelers())

    assert len(fig.axes) == 1
    assert fig.axes[0].lines
    assert fig.axes[0].collections
    assert fig.axes[0].get_title() == "direct"


def test_two_modes_get_two_panels() -> None:
    frame = travelers()
    extra = frame.iloc[[0, 1]].copy()
    extra["trip_id"] = [4, 5]
    extra["mode"] = "pnr"
    fig = plot_before_after(pd.concat([frame, extra], ignore_index=True))

    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ["direct", "pnr"]


def test_delta_has_a_panel_for_each_role() -> None:
    fig = plot_delta(travelers())

    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ["Driver", "Rider"]
    assert fig.axes[0].patches
    assert fig.axes[1].patches


def test_vehicle_title_includes_the_total() -> None:
    fig = plot_vehicle_saved(matches())

    assert fig.axes[0].get_title() == "Total 8 min saved"
    assert fig.axes[0].patches

    empty = plot_vehicle_saved(matches().iloc[0:0])
    assert empty.axes[0].get_title() == "Total 0 min saved"


def test_hours_span_the_departures() -> None:
    fig = plot_by_hour(travelers())

    assert len(fig.axes) == 2
    assert [tick.get_text() for tick in fig.axes[0].get_xticklabels()] == ["07:00", "08:00", "09:00"]
    assert fig.axes[0].patches


def test_match_map_has_before_and_after() -> None:
    fig = plot_match(0, travelers(), matches(), legs())

    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ["Before", "After"]
    assert "min saved" in fig._suptitle.get_text()

    with pytest.raises(ValueError, match="no match"):
        plot_match(4, travelers(), matches(), legs())


def test_day_map_draws_each_role() -> None:
    fig = plot_day(travelers(), matches())

    assert fig.axes[0].collections
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert labels == ["Driver", "Rider", "Solo"]
