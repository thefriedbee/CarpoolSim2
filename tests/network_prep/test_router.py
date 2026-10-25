"""Tests for shortest paths on an in-memory drive network."""

import pytest
from pyproj import Transformer

from carpoolsim.network_prep.partition import partition_network
from carpoolsim.network_prep.router import RoadRouter
from carpoolsim.network_prep.router import Route
from tests.conftest import Network


@pytest.fixture(params=[False, True])
def line_router(line_network: Network, request: pytest.FixtureRequest) -> RoadRouter:
    """Router for the six-node line, once on the full graph and once cached."""
    nodes, edges = line_network
    return RoadRouter(nodes, edges, cached=request.param, max_nodes=3)


def test_route_follows_the_line(line_router: RoadRouter) -> None:
    route = line_router.route(0, 5)

    assert isinstance(route, Route)
    assert route.travel_time == pytest.approx(5.0)
    assert route.nodes == [0, 1, 2, 3, 4, 5]
    assert route.edge_rows == [0, 2, 4, 6, 8]


def test_travel_minutes_matches_the_route(line_router: RoadRouter) -> None:
    assert line_router.travel_minutes(0, 5) == pytest.approx(5.0)
    assert line_router.travel_minutes(5, 0) == pytest.approx(5.0)
    assert line_router.travel_minutes(2, 2) == pytest.approx(0.0)
    assert line_router.route(0, 5).travel_time == pytest.approx(line_router.travel_minutes(0, 5))


def test_travel_minutes_matches_the_shortcut(shortcut_network: Network) -> None:
    nodes, edges = shortcut_network
    for cached in (False, True):
        router = RoadRouter(nodes, edges, cached=cached, max_nodes=2)
        minutes = router.travel_minutes(0, 3)
        assert minutes == pytest.approx(2.0)
        assert router.route(0, 3).travel_time == pytest.approx(minutes)


def test_same_node_is_a_zero_trip(line_router: RoadRouter) -> None:
    route = line_router.route(2, 2)

    assert route.travel_time == pytest.approx(0.0)
    assert route.nodes == [2]
    assert route.edge_rows == []


def test_shortcut_beats_the_long_way(shortcut_network: Network) -> None:
    nodes, edges = shortcut_network
    full = RoadRouter(nodes, edges, cached=False).route(0, 3)
    cached = RoadRouter(nodes, edges, cached=True, max_nodes=2).route(0, 3)

    assert full.travel_time == pytest.approx(2.0)
    assert full.nodes == [0, 1, 3]
    assert full.edge_rows == [1, 2]
    assert cached.travel_time == pytest.approx(full.travel_time)
    assert cached.nodes == full.nodes
    assert cached.edge_rows == full.edge_rows


def test_equal_travel_time_keeps_the_first_arc(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 3.0), (1, 2, 1, 3.0)],
    )

    route = RoadRouter(nodes, edges).route(0, 1)

    assert route.travel_time == pytest.approx(3.0)
    assert route.nodes == [0, 1]
    assert route.edge_rows == [0]


def test_nearest_node_uses_web_mercator(line_network: Network) -> None:
    nodes, _edges = line_network
    router = RoadRouter(nodes, _edges)
    to_lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    origin_lon, origin_lat = to_lonlat.transform(0.0, 0.0)
    near_second_lon, near_second_lat = to_lonlat.transform(900.0, 0.0)

    assert router.nearest_node(origin_lon, origin_lat) == 0
    assert router.nearest_node(near_second_lon, near_second_lat) == 1


def test_missing_path_raises(network_builder) -> None:
    nodes, edges = network_builder(
        [10, 20, 30],
        [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)],
        [(10, 20, 0, 1.0)],
    )

    full = RoadRouter(nodes, edges, cached=False)
    cached = RoadRouter(nodes, edges, cached=True, max_nodes=2)
    with pytest.raises(RuntimeError, match="No path between the selected nodes"):
        full.route(0, 2)
    with pytest.raises(RuntimeError, match="No path between the selected nodes"):
        full.travel_minutes(0, 2)
    with pytest.raises(RuntimeError, match="No path between the selected nodes"):
        cached.route(0, 2)
    with pytest.raises(RuntimeError, match="No path between the selected nodes"):
        cached.travel_minutes(0, 2)
    assert cached.travel_minutes(0, 1) == pytest.approx(cached.route(0, 1).travel_time)


def test_partition_from_other_tables_raises(line_network: Network) -> None:
    nodes, edges = line_network
    partition = partition_network(nodes, edges, max_nodes=3)

    with pytest.raises(ValueError, match="partition was built from different"):
        RoadRouter(nodes.iloc[::-1], edges, cached=True, partition=partition)


def test_cache_counts_portal_paths(line_network: Network) -> None:
    nodes, edges = line_network
    cached = RoadRouter(nodes, edges, cached=True, max_nodes=3)
    full = RoadRouter(nodes, edges, cached=False)

    assert cached.partition is not None
    assert cached.n_cached_paths > 0
    assert cached.n_overlay_arcs > 0
    assert cached.build_seconds >= 0
    assert full.partition is None
    assert full.n_cached_paths == 0
    assert full.n_overlay_arcs == 0
