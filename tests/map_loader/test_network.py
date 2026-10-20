"""Tests for travel time and GeoPackage save/load."""

import networkx as nx
import pytest

from carpoolsim.map_loader.network import add_edge_travel
from carpoolsim.map_loader.network import graph_to_network_gdfs
from carpoolsim.map_loader.network import load_network
from carpoolsim.map_loader.network import save_network
from tests.conftest import Network


def _street_graph() -> nx.MultiDiGraph:
    """Two nodes in WGS 84 with room for one street between them."""
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=0.01, y=0.0)
    return graph


def test_travel_time_uses_posted_speed() -> None:
    graph = _street_graph()
    graph.add_edge(1, 2, length=1000.0, highway="residential", maxspeed="50")

    add_edge_travel(graph)

    data = graph.edges[1, 2, 0]
    assert data["speed_kph"] == pytest.approx(50.0)
    assert data["travel_time"] == pytest.approx(1.2)


def test_travel_time_converts_miles_per_hour() -> None:
    graph = _street_graph()
    graph.add_edge(1, 2, length=1000.0, highway="residential", maxspeed="30 mph")

    add_edge_travel(graph)

    speed_kph = 30 * 1.60934
    data = graph.edges[1, 2, 0]
    assert data["speed_kph"] == pytest.approx(speed_kph)
    assert data["travel_time"] == pytest.approx(1000.0 * 60.0 / (speed_kph * 1000.0))


def test_missing_speed_uses_the_highway_mean() -> None:
    graph = _street_graph()
    graph.add_node(3, x=0.02, y=0.0)
    graph.add_edge(1, 2, length=500.0, highway="residential", maxspeed="50")
    graph.add_edge(2, 3, length=500.0, highway="residential")

    add_edge_travel(graph)

    assert graph.edges[2, 3, 0]["speed_kph"] == pytest.approx(50.0)
    assert graph.edges[2, 3, 0]["travel_time"] == pytest.approx(0.6)


def test_graph_to_network_gdfs_projects_nodes_and_keeps_travel_time() -> None:
    graph = _street_graph()
    graph.add_edge(1, 2, length=1000.0, highway="residential", maxspeed="50", osmid=[10, 11])
    graph.add_edge(2, 1, length=1000.0, highway="residential", maxspeed="50", osmid=12)

    nodes, edges = graph_to_network_gdfs(graph)

    assert list(nodes.columns[:4]) == ["lon", "lat", "x", "y"]
    assert nodes.crs.to_epsg() == 3857
    assert edges.crs.to_epsg() == 3857
    assert nodes.loc[1, "lon"] == pytest.approx(0.0)
    assert nodes.loc[1, "lat"] == pytest.approx(0.0)
    assert nodes.loc[1, "x"] == pytest.approx(0.0)
    assert nodes.loc[1, "y"] == pytest.approx(0.0)
    assert nodes.loc[2, "x"] > 0
    assert edges["travel_time"].tolist() == pytest.approx([1.2, 1.2])
    assert graph.edges[1, 2, 0]["travel_time"] == pytest.approx(1.2)


def test_save_load_round_trip(line_network: Network, tmp_path) -> None:
    nodes, edges = line_network
    edges = edges.copy()
    edges["osmid"] = [[10, 20] if i == 0 else [i] for i in range(len(edges))]

    nodes_path, edges_path = save_network(nodes, edges, tmp_path / "nested" / "net")
    loaded_nodes, loaded_edges = load_network(nodes_path.parent)

    assert nodes_path.name == "nodes.gpkg"
    assert edges_path.name == "edges.gpkg"
    assert loaded_nodes.index.names == ["osmid"]
    assert loaded_edges.index.names == ["u", "v", "key"]
    assert loaded_nodes.index.tolist() == nodes.index.tolist()
    assert loaded_edges.index.tolist() == edges.index.tolist()
    assert loaded_nodes["x"].tolist() == pytest.approx(nodes["x"].tolist())
    assert loaded_nodes["y"].tolist() == pytest.approx(nodes["y"].tolist())
    assert loaded_edges["travel_time"].tolist() == pytest.approx(edges["travel_time"].tolist())
    assert loaded_edges["osmid"].iloc[0] == "10,20"
    assert loaded_edges["osmid"].iloc[1] == "1"
    assert isinstance(edges["osmid"].iloc[0], list)
