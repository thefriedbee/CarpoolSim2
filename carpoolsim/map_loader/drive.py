"""Load an OSM XML file and keep drivable streets.

``graph_from_xml`` keeps every OSM way, including land-use outlines, paths,
and footways. :func:`keep_drivable_edges` applies the same street filter as
OSMnx ``network_type="drive"``.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import osmnx as ox

# Highway values excluded by osmnx ``network_type="drive"``.
EXCLUDED_HIGHWAYS = frozenset(
    {
        "abandoned",
        "bridleway",
        "bus_guideway",
        "construction",
        "corridor",
        "cycleway",
        "elevator",
        "escalator",
        "footway",
        "no",
        "path",
        "pedestrian",
        "planned",
        "platform",
        "proposed",
        "raceway",
        "razed",
        "rest_area",
        "service",
        "services",
        "steps",
        "track",
    }
)

# ``service=*`` values excluded by osmnx ``network_type="drive"``.
EXCLUDED_SERVICE = frozenset(
    {
        "alley",
        "driveway",
        "emergency_access",
        "parking",
        "parking_aisle",
        "private",
    }
)


def _tag_values(value: object) -> list[object]:
    """Return one OSM tag value as a list.

    Simplified edges sometimes store a tag as a list of the merged values.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def is_drivable(edge_data: dict[str, object]) -> bool:
    """Return whether one edge is a public drivable street.

    An edge is kept when it has a ``highway`` tag and is not an area polygon,
    a private way, a non-drive highway type, or a dropped service type such as
    a driveway or parking aisle.
    """
    highways = _tag_values(edge_data.get("highway"))
    if not highways or any(highway in EXCLUDED_HIGHWAYS for highway in highways):
        return False
    if edge_data.get("area") == "yes" or edge_data.get("access") == "private":
        return False
    if any(service in EXCLUDED_SERVICE for service in _tag_values(edge_data.get("service"))):
        return False
    if edge_data.get("motor_vehicle") == "no" or edge_data.get("motorcar") == "no":
        return False
    return True


def load_osm_graph(
    filepath: str | Path,
    *,
    bidirectional: bool = False,
    simplify: bool = True,
    retain_all: bool = False,
) -> nx.MultiDiGraph:
    """Load a street graph from an OSM XML file.

    ``filepath`` must be OSM XML (``.osm`` or ``.osm.bz2``), not a ``.pbf``
    extract. When ``retain_all`` is false, the largest strongly connected
    component is returned.
    """
    graph = ox.graph_from_xml(
        filepath,
        bidirectional=bidirectional,
        simplify=simplify,
        retain_all=retain_all,
    )
    if not retain_all:
        graph = ox.truncate.largest_component(graph, strongly=True)
    return graph


def keep_drivable_edges(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Drop non-drivable edges from ``graph`` and return it.

    The graph is updated in place. Isolated nodes are removed, then the
    largest strongly connected component is kept.
    """
    dropped = [
        (u, v, key)
        for u, v, key, data in graph.edges(keys=True, data=True)
        if not is_drivable(data)
    ]
    graph.remove_edges_from(dropped)
    graph.remove_nodes_from(node for node in list(graph.nodes) if graph.degree(node) == 0)
    return ox.truncate.largest_component(graph, strongly=True)
