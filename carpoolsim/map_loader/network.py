"""Convert a street graph to node and edge GeoDataFrames and store them.

Node rows keep longitude and latitude plus ``x`` and ``y`` in EPSG:3857.
Edge lines are stored in EPSG:3857, with free-flow ``speed_kph`` and
``travel_time`` in minutes. Both tables are written as GeoPackages so a
later session can reload them without parsing the OSM file again.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox

from carpoolsim.map_loader.drive import keep_drivable_edges
from carpoolsim.map_loader.drive import load_osm_graph

NODES_FILENAME = "nodes.gpkg"
EDGES_FILENAME = "edges.gpkg"
WEB_MERCATOR = 3857


def add_edge_travel(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Add free-flow speed and travel time to each edge of ``graph``.

    ``speed_kph`` is the posted ``maxspeed`` in kilometers per hour. Values
    tagged in miles per hour are converted. An edge with no usable limit gets
    the mean ``speed_kph`` of its ``highway`` type. ``travel_time`` is the
    free-flow time in minutes, from ``length`` in meters and ``speed_kph``.

    The graph is updated in place.
    """
    ox.add_edge_speeds(graph)
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        data["travel_time"] = float(data["length"]) * 60.0 / (float(data["speed_kph"]) * 1000.0)
    return graph


def graph_to_network_gdfs(
    graph: nx.MultiDiGraph,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build node and edge GeoDataFrames in EPSG:3857.

    ``nodes`` is indexed by OSM id and has ``lon``, ``lat``, ``x``, and ``y``.
    ``lon`` and ``lat`` are the original WGS 84 coordinates. ``x`` and ``y``
    are the same points in EPSG:3857 meters, and the point geometry uses that
    projection.

    ``edges`` is indexed by ``(u, v, key)`` and its line geometry is also in
    EPSG:3857. Each edge has ``speed_kph`` and ``travel_time`` in minutes,
    added with :func:`add_edge_travel`.
    """
    add_edge_travel(graph)
    nodes = ox.graph_to_gdfs(graph, edges=False)
    nodes["lon"] = nodes["x"]
    nodes["lat"] = nodes["y"]
    nodes = nodes.drop(columns=["x", "y"]).to_crs(epsg=WEB_MERCATOR)
    nodes["x"] = nodes.geometry.x
    nodes["y"] = nodes.geometry.y
    front = ["lon", "lat", "x", "y"]
    nodes = nodes[front + [column for column in nodes.columns if column not in front]]

    edges = ox.graph_to_gdfs(graph, nodes=False).to_crs(epsg=WEB_MERCATOR)
    return nodes, edges


def _gdf_for_file(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a copy that a GeoPackage can store.

    Index names such as ``osmid`` or ``u``, ``v``, and ``key`` become columns.
    List-valued tags, including a simplified edge's ``osmid``, are written as
    comma-separated text.
    """
    out = gdf.reset_index()
    geometry_name = out.active_geometry_name
    for column in out.columns:
        if column == geometry_name:
            continue
        has_list = out[column].map(lambda value: isinstance(value, (list, tuple, set))).any()
        if has_list:
            out[column] = out[column].map(
                lambda value: (
                    ",".join(map(str, value)) if isinstance(value, (list, tuple, set)) else value
                )
            )
    return out


def save_network(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    directory: str | Path,
) -> tuple[Path, Path]:
    """Write ``nodes.gpkg`` and ``edges.gpkg`` under ``directory``.

    Returns the two file paths. The in-memory frames are not modified.
    """
    network_dir = Path(directory)
    network_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = network_dir / NODES_FILENAME
    edges_path = network_dir / EDGES_FILENAME
    _gdf_for_file(nodes).to_file(nodes_path, driver="GPKG")
    _gdf_for_file(edges).to_file(edges_path, driver="GPKG")
    return nodes_path, edges_path


def load_network(directory: str | Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Reload node and edge GeoDataFrames written by :func:`save_network`.

    ``nodes`` is indexed by ``osmid``. ``edges`` is indexed by ``(u, v, key)``.
    List-valued tags come back as comma-separated text.
    """
    network_dir = Path(directory)
    nodes = gpd.read_file(network_dir / NODES_FILENAME)
    edges = gpd.read_file(network_dir / EDGES_FILENAME)
    if "osmid" in nodes.columns:
        nodes = nodes.set_index("osmid")
    if {"u", "v", "key"} <= set(edges.columns):
        edges = edges.set_index(["u", "v", "key"])
    return nodes, edges


def build_drive_network(
    filepath: str | Path,
    directory: str | Path | None = None,
) -> tuple[nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load an OSM XML file, keep drivable streets, and build both tables.

    Edge rows include free-flow ``speed_kph`` and ``travel_time`` in minutes.
    When ``directory`` is set, the tables are also written there with
    :func:`save_network`. For streets and parking lots from the same file,
    use :func:`carpoolsim.map_loader.lots.build_map`.
    """
    graph = keep_drivable_edges(load_osm_graph(filepath))
    nodes, edges = graph_to_network_gdfs(graph)
    if directory is not None:
        save_network(nodes, edges, directory)
    return graph, nodes, edges
