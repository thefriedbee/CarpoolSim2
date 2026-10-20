"""Individual-trip maps use saved results without routing a new journey."""

import matplotlib

matplotlib.use("Agg", force=True)

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from pandas.testing import assert_frame_equal
from shapely.geometry import LineString

from evaluation import plot_trip
from evaluation.maps import DRIVER_COLOR
from evaluation.maps import RIDER_COLOR


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def result_tables():
    """A carpool, a curved solo path, and a distant unrelated trip."""
    travelers = pd.DataFrame(
        {
            "trip_id": [10, 20, 30, 40],
            "role": ["driver", "rider", "solo", "solo"],
            "mode": ["direct", "direct", "", ""],
            "solo_min": [20.0, 15.0, 30.0, 10.0],
            "carpool_min": [24.0, 18.0, 30.0, 10.0],
            "wait_min": [0.0, 3.0, 0.0, 0.0],
            "access_min": [0.0, 0.0, 0.0, 0.0],
            "delta_min": [4.0, 3.0, 0.0, 0.0],
            "origin_x": [0.0, 1000.0, 500.0, 1000000.0],
            "origin_y": [0.0, 0.0, 1000.0, 1000000.0],
            "dest_x": [4000.0, 3000.0, 1500.0, 1001000.0],
            "dest_y": [0.0, 0.0, 1000.0, 1000000.0],
        }
    )
    matches = pd.DataFrame(
        {
            "match_id": [7],
            "driver_id": [10],
            "rider_id": [20],
            "mode": ["direct"],
            "lot_x": [float("nan")],
            "lot_y": [float("nan")],
            "vehicle_min_saved": [8.0],
        }
    )
    legs = gpd.GeoDataFrame(
        {
            "trip_id": pd.array([10, 20, 10, 10, 10, 30, 40], dtype="Int64"),
            "match_id": pd.array([7, 7, 7, 7, 7, None, None], dtype="Int64"),
            "who": ["driver", "rider", "driver", "driver", "driver", "solo", "solo"],
            "leg": ["solo", "solo", "pickup", "shared", "dropoff", "solo", "solo"],
        },
        geometry=[
            LineString([(0, 0), (2000, 250), (4000, 0)]),
            LineString([(1000, 0), (2000, -250), (3000, 0)]),
            LineString([(0, 0), (500, 50), (1000, 0)]),
            LineString([(1000, 0), (2000, 100), (3000, 0)]),
            LineString([(3000, 0), (3500, 50), (4000, 0)]),
            LineString([(500, 1000), (1000, 2500), (1500, 1000)]),
            LineString([(1000000, 1000000), (1001000, 1000000)]),
        ],
        crs="EPSG:3857",
    )
    return travelers, matches, legs


def line_segments(ax):
    """Read plotted route coordinates, excluding scatter marker outlines."""
    segments = [line.get_xydata() for line in ax.lines]
    for collection in ax.collections:
        if isinstance(collection, LineCollection):
            segments.extend(collection.get_segments())
    return segments


def has_route(ax, geometry):
    expected = np.asarray(geometry.coords)
    return any(
        segment.shape == expected.shape and np.allclose(segment, expected)
        for segment in line_segments(ax)
    )


def legend_labels(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()]


def route_is_dashed(collection):
    pattern = collection.get_linestyles()[0][1]
    return pattern is not None


def routes_of(ax, color):
    target = np.round(to_rgba(color), 5)
    return [
        collection
        for collection in ax.collections
        if isinstance(collection, LineCollection)
        and len(collection.get_edgecolors())
        and np.allclose(np.round(collection.get_edgecolors()[0], 5), target)
    ]


def test_solo_uses_its_curved_stored_route_without_other_trips_or_mutation(result_tables):
    travelers, matches, legs = result_tables
    before = [frame.copy(deep=True) for frame in result_tables]

    fig = plot_trip(30, travelers, matches, legs)

    assert [ax.get_title() for ax in fig.axes] == ["Drive alone"]
    ax = fig.axes[0]
    assert len(line_segments(ax)) == 1
    assert has_route(ax, legs.geometry.iloc[5])
    assert ax.get_xlim()[1] < 100000
    assert ax.get_ylim()[1] >= 2500
    assert_frame_equal(travelers, before[0])
    assert_frame_equal(matches, before[1])
    assert_geodataframe_equal(legs, before[2])


@pytest.mark.parametrize("trip_id,role", [(10, "driver"), (20, "rider")])
def test_direct_trip_finds_match_from_either_member_and_highlights_selection(result_tables, trip_id, role):
    travelers, matches, legs = result_tables

    fig = plot_trip(trip_id, travelers, matches, legs)

    assert [ax.get_title() for ax in fig.axes] == ["Drive alone baseline", "Direct carpool"]
    assert legend_labels(fig.axes[0]) == ["Origin", "Destination", "Driver drives", "Rider drives"]
    assert legend_labels(fig.axes[1]) == ["Origin", "Destination", "Driver drives", "Shared"]
    after = fig.axes[1]
    assert routes_of(after, RIDER_COLOR) == []
    assert routes_of(after, DRIVER_COLOR)
    assert any(route_is_dashed(route) for route in routes_of(after, DRIVER_COLOR))
    assert any(not route_is_dashed(route) for route in routes_of(after, DRIVER_COLOR))
    assert len(line_segments(fig.axes[0])) == 2
    for index in (0, 1):
        assert has_route(fig.axes[0], legs.geometry.iloc[index])
    for ax in fig.axes:
        assert any(text.get_text() == f"Trip {trip_id}" for text in ax.texts)
        assert ax.get_xlim()[1] < 100000
    for index in (2, 3, 4):
        assert has_route(fig.axes[1], legs.geometry.iloc[index])
    assert not has_route(fig.axes[1], legs.geometry.iloc[6])
    title = fig._suptitle.get_text().lower()
    assert f"trip {trip_id}" in title
    assert role in title
    assert all(name in title for name in ("solo", "carpool", "wait", "access"))


def test_pnr_is_identified_from_lot_and_draws_saved_access_even_for_custom_mode(result_tables):
    travelers, matches, legs = result_tables
    travelers.loc[travelers["trip_id"].isin([10, 20]), "mode"] = "airport express"
    travelers.loc[travelers["trip_id"] == 20, "access_min"] = 2.0
    matches.loc[0, ["mode", "lot_x", "lot_y"]] = ["airport express", 2000.0, 500.0]
    access = LineString([(1000, 0), (1400, 350), (2000, 500)])
    legs = pd.concat(
        [
            legs,
            gpd.GeoDataFrame(
                {"trip_id": [20], "match_id": [7], "who": ["rider"], "leg": ["access"]},
                geometry=[access],
                crs=legs.crs,
            ),
        ],
        ignore_index=True,
    )

    fig = plot_trip(20, travelers, matches, legs)

    assert [ax.get_title() for ax in fig.axes] == ["Drive alone baseline", "PNR carpool"]
    assert has_route(fig.axes[1], access)
    assert legend_labels(fig.axes[1]) == [
        "Origin",
        "Destination",
        "PNR facility",
        "Driver drives",
        "Rider drives",
        "Shared",
    ]
    lot_markers = [
        collection for collection in fig.axes[1].collections if collection.get_label() == "PNR facility"
    ]
    assert len(lot_markers) == 1
    np.testing.assert_allclose(lot_markers[0].get_offsets(), [[2000, 500]])
    assert any(text.get_text() == "P" for text in fig.axes[1].texts)
    assert routes_of(fig.axes[1], RIDER_COLOR)
    assert all(not route_is_dashed(route) for route in routes_of(fig.axes[1], RIDER_COLOR))
    assert any(route_is_dashed(route) for route in routes_of(fig.axes[1], DRIVER_COLOR))


@pytest.mark.parametrize("missing_geometry", [None, LineString()])
def test_unusable_solo_geometry_uses_labelled_fallback_with_finite_zero_length_bounds(
    result_tables, missing_geometry
):
    travelers, matches, legs = result_tables
    travelers.loc[travelers["trip_id"] == 30, ["dest_x", "dest_y"]] = [500.0, 1000.0]
    legs.loc[legs["trip_id"] == 30, "geometry"] = missing_geometry

    fig = plot_trip(30, travelers, matches, legs)

    ax = fig.axes[0]
    assert "Straight-line fallback" in legend_labels(ax)
    assert any("road route" in text.get_text().lower() for text in ax.texts)
    assert has_route(ax, LineString([(500, 1000), (500, 1000)]))
    assert np.isfinite([*ax.get_xlim(), *ax.get_ylim()]).all()
    assert ax.get_xlim()[0] < ax.get_xlim()[1]
    assert ax.get_ylim()[0] < ax.get_ylim()[1]


def test_legacy_outputs_without_trip_ids_use_dashed_solo_fallback(result_tables):
    travelers, matches, legs = result_tables
    legacy_legs = legs.iloc[:5].drop(columns="trip_id")

    fig = plot_trip(30, travelers, matches, legacy_legs)

    ax = fig.axes[0]
    assert len(line_segments(ax)) == 1
    assert has_route(ax, LineString([(500, 1000), (1500, 1000)]))
    assert "Straight-line fallback" in legend_labels(ax)
    assert any("road route" in text.get_text().lower() for text in ax.texts)
    fallback_lines = [line for line in ax.lines if line.get_label() == "Straight-line fallback"]
    fallback_collections = [
        collection
        for collection in ax.collections
        if isinstance(collection, LineCollection) and collection.get_label() == "Straight-line fallback"
    ]
    assert any(line.get_linestyle() == "--" for line in fallback_lines) or any(
        dash_pattern is not None
        for collection in fallback_collections
        for _, dash_pattern in collection.get_linestyles()
    )


def test_unknown_trip_or_inconsistent_match_fails_clearly(result_tables):
    travelers, matches, legs = result_tables
    with pytest.raises(ValueError):
        plot_trip(999, travelers, matches, legs)
    with pytest.raises(ValueError):
        plot_trip(10, travelers, matches.iloc[0:0], legs)
    with pytest.raises(ValueError):
        plot_trip(10, travelers, pd.concat([matches, matches], ignore_index=True), legs)
    with pytest.raises(ValueError):
        plot_trip(10, travelers[travelers["trip_id"] != 20], matches, legs)
    with pytest.raises(ValueError):
        plot_trip(10, pd.concat([travelers, travelers.iloc[[0]]], ignore_index=True), matches, legs)
    wrong_role = travelers.copy()
    wrong_role.loc[wrong_role["trip_id"] == 10, "role"] = "rider"
    with pytest.raises(ValueError):
        plot_trip(10, wrong_role, matches, legs)
