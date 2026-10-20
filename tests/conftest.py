"""Shared in-memory drive networks."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

Arc = tuple[int, int, int, float]
Network = tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]


def build_network(
    osmids: Sequence[int],
    coords: Sequence[tuple[float, float]],
    arcs: Sequence[Arc],
) -> Network:
    """Return node and edge tables for a tiny drive network.

    ``coords`` are ``(x, y)`` in EPSG:3857 meters, one per OSM id.
    ``arcs`` are ``(u, v, key, travel_time)`` with ``u`` and ``v`` OSM ids.
    Node row ``i`` is ``osmids[i]``. Edge row ``i`` is ``arcs[i]``.
    """
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]
    nodes = gpd.GeoDataFrame(
        {"x": x_coords, "y": y_coords},
        geometry=gpd.points_from_xy(x_coords, y_coords),
        crs="EPSG:3857",
        index=pd.Index(osmids, name="osmid"),
    )
    coord_of = dict(zip(osmids, coords, strict=True))
    rows = []
    index = []
    for start, end, key, minutes in arcs:
        rows.append(
            {
                "travel_time": minutes,
                "geometry": LineString([coord_of[start], coord_of[end]]),
            }
        )
        index.append((start, end, key))
    edges = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:3857",
        index=pd.MultiIndex.from_tuples(index, names=["u", "v", "key"]),
    )
    return nodes, edges


@pytest.fixture
def network_builder() -> Callable[
    [Sequence[int], Sequence[tuple[float, float]], Sequence[Arc]],
    Network,
]:
    """Return the in-memory network builder."""
    return build_network


@pytest.fixture
def line_network(network_builder: Callable[..., Network]) -> Network:
    """Six nodes on a line. Each hop is one minute, forward edge then back."""
    count = 6
    osmids = list(range(101, 101 + count))
    coords = [(float(i * 1000), 0.0) for i in range(count)]
    arcs: list[Arc] = []
    for i in range(count - 1):
        arcs.append((osmids[i], osmids[i + 1], 0, 1.0))
        arcs.append((osmids[i + 1], osmids[i], 0, 1.0))
    return network_builder(osmids, coords, arcs)


@pytest.fixture
def shortcut_network(network_builder: Callable[..., Network]) -> Network:
    """Two ways from node 1 to node 4. The path through node 2 is faster.

    The first arc from 1 to 2 is slower than the second, so a shortest path
    must take the second arc.
    """
    osmids = [1, 2, 3, 4]
    coords = [(0.0, 0.0), (1000.0, 1000.0), (1000.0, -1000.0), (2000.0, 0.0)]
    arcs = [
        (1, 2, 0, 5.0),
        (1, 2, 1, 1.0),
        (2, 4, 0, 1.0),
        (1, 3, 0, 4.0),
        (3, 4, 0, 4.0),
        (2, 1, 0, 9.0),
        (4, 2, 0, 9.0),
        (3, 1, 0, 9.0),
        (4, 3, 0, 9.0),
    ]
    return network_builder(osmids, coords, arcs)


@pytest.fixture(params=["line", "shortcut"])
def split_network(
    request: pytest.FixtureRequest,
    line_network: Network,
    shortcut_network: Network,
) -> Network:
    """A network large enough to split when ``max_nodes`` is 3."""
    if request.param == "line":
        return line_network
    return shortcut_network
