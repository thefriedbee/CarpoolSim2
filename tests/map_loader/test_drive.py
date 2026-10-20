"""Tests for drivable-street filtering."""

import networkx as nx
import pytest

from carpoolsim.map_loader.drive import EXCLUDED_HIGHWAYS
from carpoolsim.map_loader.drive import EXCLUDED_SERVICE
from carpoolsim.map_loader.drive import is_drivable
from carpoolsim.map_loader.drive import keep_drivable_edges


@pytest.mark.parametrize("highway", sorted(EXCLUDED_HIGHWAYS))
def test_excluded_highway_is_not_drivable(highway: str) -> None:
    assert is_drivable({"highway": highway}) is False


@pytest.mark.parametrize("service", sorted(EXCLUDED_SERVICE))
def test_excluded_service_is_not_drivable(service: str) -> None:
    assert is_drivable({"highway": "residential", "service": service}) is False
    assert is_drivable({"highway": "residential", "service": [service]}) is False


@pytest.mark.parametrize(
    ("edge", "expected"),
    [
        ({"highway": "residential"}, True),
        ({"highway": ["primary", "secondary"]}, True),
        ({"highway": ["residential", "footway"]}, False),
        ({}, False),
        ({"highway": "residential", "area": "yes"}, False),
        ({"highway": "residential", "access": "private"}, False),
        ({"highway": "residential", "access": "permissive"}, True),
        ({"highway": "residential", "motor_vehicle": "no"}, False),
        ({"highway": "residential", "motorcar": "no"}, False),
        ({"highway": "residential", "service": "yes"}, True),
    ],
)
def test_is_drivable(edge: dict[str, object], expected: bool) -> None:
    assert is_drivable(edge) is expected


def test_keep_drivable_edges_drops_other_ways() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, highway="residential")
    graph.add_edge(2, 1, highway="residential")
    graph.add_edge(1, 2, key=1, highway="footway")
    graph.add_edge(2, 3, highway="footway")
    graph.add_edge(3, 2, highway="footway")
    graph.add_edge(2, 4, highway=["residential", "path"])
    graph.add_edge(4, 2, highway=["residential", "path"])

    kept = keep_drivable_edges(graph)

    assert kept is graph
    assert set(kept.nodes) == {1, 2}
    assert {(u, v, key) for u, v, key in kept.edges(keys=True)} == {(1, 2, 0), (2, 1, 0)}
    assert all(is_drivable(data) for _, _, data in kept.edges(data=True))
