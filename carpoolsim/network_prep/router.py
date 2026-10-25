"""Shortest paths on a drive network.

``RoadRouter`` searches the whole network with Dijkstra. Set ``cached=True``
to store shortest paths between the portals of each part and search that
overlay instead. ``travel_minutes`` returns that time alone.
``route`` also returns the nodes and the edge rows. Edge data on every
graph is free-flow travel time in minutes, and ``float`` is the weight
function so Dijkstra reads that time.
"""

from __future__ import annotations

import time

import geopandas as gpd
import numpy as np
import rustworkx as rx
from pyproj import Transformer

from carpoolsim.network_prep.partition import NetworkPartition
from carpoolsim.network_prep.partition import endpoints
from carpoolsim.network_prep.partition import partition_network


class Route:
    """One shortest path.

    ``travel_time`` is the free-flow time in minutes. ``nodes`` and
    ``edge_rows`` are row positions in the node table and the edge table.
    """

    def __init__(self, travel_time: float, nodes: list[int], edge_rows: list[int]) -> None:
        self.travel_time = travel_time
        self.nodes = nodes
        self.edge_rows = edge_rows


class RoadRouter:
    """Drive-network router, with an optional portal cache.

    Node arguments are row positions in the node table used to build the
    router. When ``cached`` is true, pass the ``partition`` built from those
    same tables, or leave it empty and one is built here.
    """

    def __init__(
        self,
        nodes: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        *,
        cached: bool = False,
        partition: NetworkPartition | None = None,
        max_nodes: int = 400,
        min_share: float = 0.30,
    ) -> None:
        xy, src, dst = endpoints(nodes, edges)
        if partition is None and cached:
            partition = partition_network(nodes, edges, max_nodes=max_nodes, min_share=min_share)
        if partition is not None and (
            not np.array_equal(partition.src, src) or not np.array_equal(partition.dst, dst)
        ):
            raise ValueError("partition was built from different node or edge tables")

        self.cached = cached
        self.partition = partition
        self._xy = xy
        self._best_arc = _shortest_arcs(src, dst, edges["travel_time"].to_numpy())
        self._to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.n_cached_paths = 0
        self.n_overlay_arcs = 0
        self.build_seconds = 0.0
        self._road: rx.PyDiGraph | None = None

        if cached:
            started = time.perf_counter()
            self._build_cache()
            self.build_seconds = time.perf_counter() - started
        else:
            road = rx.PyDiGraph()
            road.add_nodes_from(range(len(xy)))
            road.add_edges_from(
                (u, v, minutes) for (u, v), (minutes, _row) in self._best_arc.items()
            )
            self._road = road

    def nearest_node(self, lon: float, lat: float) -> int:
        """Return the node row closest to a WGS 84 point."""
        x, y = self._to_mercator.transform(lon, lat)
        offset = self._xy - np.array([x, y])
        return int((offset[:, 0] ** 2 + offset[:, 1] ** 2).argmin())

    def route(self, origin: int, destination: int) -> Route:
        """Return a shortest path from ``origin`` to ``destination``."""
        origin = int(origin)
        destination = int(destination)
        if self.cached:
            return self._route_cached(origin, destination)
        return self._route_full(origin, destination)

    def travel_minutes(self, origin: int, destination: int) -> float:
        """Return the free-flow minutes from ``origin`` to ``destination``.

        The time matches :meth:`route`. On the full street graph the search
        stops at the destination and does not build the node path. A missing
        path raises ``RuntimeError``.
        """
        origin = int(origin)
        destination = int(destination)
        if origin == destination:
            return 0.0
        if self.cached:
            return self._minutes_cached(origin, destination)
        return self._minutes_full(origin, destination)

    def _route_full(self, origin: int, destination: int) -> Route:
        if origin == destination:
            return Route(0.0, [origin], [])
        assert self._road is not None
        found = rx.dijkstra_shortest_paths(self._road, origin, destination, weight_fn=float)
        if destination not in found:
            raise RuntimeError("No path between the selected nodes.")
        path = [int(node) for node in found[destination]]
        return self._route_from_path(path)

    def _minutes_full(self, origin: int, destination: int) -> float:
        """Free-flow minutes on the whole street graph."""
        assert self._road is not None
        lengths = rx.dijkstra_shortest_path_lengths(
            self._road, origin, float, goal=destination
        )
        if destination not in lengths:
            raise RuntimeError("No path between the selected nodes.")
        return float(lengths[destination])

    def _build_cache(self) -> None:
        """Store portal-to-portal paths and the overlay that joins them."""
        partition = self.partition
        assert partition is not None
        part_arcs: list[list[tuple[int, int, float]]] = [[] for _ in partition.parts]
        cut_arcs: list[tuple[int, int, float]] = []
        for (u, v), (minutes, _row) in self._best_arc.items():
            cu = int(partition.cell_of[u])
            cv = int(partition.cell_of[v])
            if cu == cv:
                part_arcs[cu].append((u, v, minutes))
            else:
                cut_arcs.append((u, v, minutes))

        self._forward = []
        self._reverse = []
        self._local_of = []
        self._global_of = []
        self._portals_of = []
        for cid, members in enumerate(partition.parts):
            member_ids = [int(node) for node in members.tolist()]
            loc = {node: i for i, node in enumerate(member_ids)}
            forward = rx.PyDiGraph()
            reverse = rx.PyDiGraph()
            forward.add_nodes_from(member_ids)
            reverse.add_nodes_from(member_ids)
            forward.add_edges_from((loc[u], loc[v], minutes) for u, v, minutes in part_arcs[cid])
            reverse.add_edges_from((loc[v], loc[u], minutes) for u, v, minutes in part_arcs[cid])
            self._forward.append(forward)
            self._reverse.append(reverse)
            self._local_of.append(loc)
            self._global_of.append(member_ids)
            self._portals_of.append([node for node in member_ids if partition.is_portal[node]])

        self._portal_path: dict[int, dict[int, tuple[float, list[int]]]] = {}
        cached = 0
        for cid, portals in enumerate(self._portals_of):
            for portal in portals:
                reached = self._search_forward(cid, portal)
                bucket = {}
                for other in portals:
                    if other != portal and other in reached:
                        bucket[other] = reached[other]
                        cached += 1
                self._portal_path[portal] = bucket
        self.n_cached_paths = cached

        overlay = rx.PyDiGraph()
        portal_nodes = [int(portal) for portal in partition.portal_ids.tolist()]
        overlay_ids = list(overlay.add_nodes_from(portal_nodes))
        self._overlay_of = {node: idx for node, idx in zip(portal_nodes, overlay_ids)}
        overlay_arcs = []
        for src_portal, dests in self._portal_path.items():
            src_i = self._overlay_of[src_portal]
            for dst_portal, (minutes, _path) in dests.items():
                overlay_arcs.append((src_i, self._overlay_of[dst_portal], minutes))
        for u, v, minutes in cut_arcs:
            overlay_arcs.append((self._overlay_of[u], self._overlay_of[v], minutes))
        overlay.add_edges_from(overlay_arcs)
        self._overlay = overlay
        self._super_src = overlay.add_node(None)
        self._super_sink = overlay.add_node(None)
        self.n_overlay_arcs = len(overlay_arcs)

    def _search_forward(self, cid: int, source: int) -> dict[int, tuple[float, list[int]]]:
        """Shortest paths from ``source`` to nodes reached inside this part."""
        paths = rx.dijkstra_shortest_paths(
            self._forward[cid], self._local_of[cid][source], weight_fn=float
        )
        out = {source: (0.0, [source])}
        for local_path in paths.values():
            path = [self._global_of[cid][i] for i in local_path]
            out[path[-1]] = (self._travel_time_of(path), path)
        return out

    def _search_reverse(self, cid: int, source: int) -> dict[int, tuple[float, list[int]]]:
        """Shortest paths from portals inside this part to ``source``.

        The reversed graph points toward ``source``, so one search reaches
        every portal that can drive there without leaving the part.
        """
        paths = rx.dijkstra_shortest_paths(
            self._reverse[cid], self._local_of[cid][source], weight_fn=float
        )
        out = {source: (0.0, [source])}
        for local_path in paths.values():
            path = [self._global_of[cid][i] for i in local_path]
            path.reverse()
            out[path[0]] = (self._travel_time_of(path), path)
        return out

    def _route_cached(self, origin: int, destination: int) -> Route:
        """Search the two end parts and join them with cached portal paths."""
        if origin == destination:
            return Route(0.0, [origin], [])
        partition = self.partition
        assert partition is not None
        origin_cell = int(partition.cell_of[origin])
        destination_cell = int(partition.cell_of[destination])
        reached = self._search_forward(origin_cell, origin)
        arrived = self._search_reverse(destination_cell, destination)
        # A trip that never leaves its part is settled by the search above.
        direct = (
            reached[destination]
            if origin_cell == destination_cell and destination in reached
            else None
        )

        starts = {
            portal: reached[portal]
            for portal in self._portals_of[origin_cell]
            if portal in reached
        }
        ends = {
            portal: arrived[portal]
            for portal in self._portals_of[destination_cell]
            if portal in arrived
        }
        edge_ids = []
        try:
            for portal, (minutes, _path) in starts.items():
                edge_ids.append(
                    self._overlay.add_edge(self._super_src, self._overlay_of[portal], minutes)
                )
            for portal, (minutes, _path) in ends.items():
                edge_ids.append(
                    self._overlay.add_edge(self._overlay_of[portal], self._super_sink, minutes)
                )
            found = rx.dijkstra_shortest_paths(
                self._overlay, self._super_src, self._super_sink, weight_fn=float
            )
        finally:
            for edge_id in reversed(edge_ids):
                self._overlay.remove_edge_from_index(edge_id)

        via = None
        if self._super_sink in found:
            portals = [self._overlay[i] for i in found[self._super_sink][1:-1]]
            node_path = list(starts[portals[0]][1])
            for start, end in zip(portals, portals[1:]):
                if partition.cell_of[start] == partition.cell_of[end]:
                    _minutes, middle = self._portal_path[start][end]
                    node_path.extend(middle[1:])
                else:
                    node_path.append(end)
            node_path.extend(ends[portals[-1]][1][1:])
            via = node_path

        options = []
        if direct is not None:
            options.append(direct[1])
        if via is not None:
            options.append(via)
        if not options:
            raise RuntimeError("No path between the selected nodes.")
        path = min(options, key=self._travel_time_of)
        return self._route_from_path(path)

    def _minutes_cached(self, origin: int, destination: int) -> float:
        """Free-flow minutes through the portal overlay, without a node path."""
        partition = self.partition
        assert partition is not None
        origin_cell = int(partition.cell_of[origin])
        destination_cell = int(partition.cell_of[destination])
        reached = self._part_minutes(self._forward[origin_cell], origin_cell, origin)
        arrived = self._part_minutes(self._reverse[destination_cell], destination_cell, destination)
        direct = (
            reached[destination]
            if origin_cell == destination_cell and destination in reached
            else None
        )
        starts = {
            portal: reached[portal]
            for portal in self._portals_of[origin_cell]
            if portal in reached
        }
        ends = {
            portal: arrived[portal]
            for portal in self._portals_of[destination_cell]
            if portal in arrived
        }
        via = self._overlay_minutes(starts, ends)
        options = [minutes for minutes in (direct, via) if minutes is not None]
        if not options:
            raise RuntimeError("No path between the selected nodes.")
        return min(options)

    def _part_minutes(self, graph: rx.PyDiGraph, cid: int, source: int) -> dict[int, float]:
        """Minutes from ``source`` to nodes reached inside this part.

        Keys are node rows in the table used to build the router. ``source``
        is included at zero minutes.
        """
        lengths = rx.dijkstra_shortest_path_lengths(
            graph, self._local_of[cid][source], float
        )
        global_of = self._global_of[cid]
        reached = {source: 0.0}
        for local_index, minutes in lengths.items():
            reached[global_of[int(local_index)]] = float(minutes)
        return reached

    def _overlay_minutes(self, starts: dict[int, float], ends: dict[int, float]) -> float | None:
        """Minutes from the origin-side portals to the destination-side portals.

        Temporary edges are removed before this returns, including when the
        search fails. ``None`` means the two sides do not meet.
        """
        if not starts or not ends:
            return None
        edge_ids: list[int] = []
        try:
            for portal, minutes in starts.items():
                edge_ids.append(
                    self._overlay.add_edge(self._super_src, self._overlay_of[portal], minutes)
                )
            for portal, minutes in ends.items():
                edge_ids.append(
                    self._overlay.add_edge(self._overlay_of[portal], self._super_sink, minutes)
                )
            found = rx.dijkstra_shortest_path_lengths(
                self._overlay, self._super_src, float, goal=self._super_sink
            )
        finally:
            for edge_id in reversed(edge_ids):
                self._overlay.remove_edge_from_index(edge_id)
        if self._super_sink not in found:
            return None
        return float(found[self._super_sink])

    def _travel_time_of(self, path: list[int]) -> float:
        """Return the free-flow travel time of a node path, in minutes."""
        total = 0.0
        for start, end in zip(path, path[1:]):
            total += self._best_arc[(start, end)][0]
        return total

    def _route_from_path(self, path: list[int]) -> Route:
        rows = [self._best_arc[(start, end)][1] for start, end in zip(path, path[1:])]
        return Route(self._travel_time_of(path), path, rows)


def _shortest_arcs(
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray,
) -> dict[tuple[int, int], tuple[float, int]]:
    """Keep the faster directed street between each ordered pair of nodes."""
    best: dict[tuple[int, int], tuple[float, int]] = {}
    for row, u, v, weight in zip(range(len(src)), src.tolist(), dst.tolist(), weights.tolist()):
        key = (int(u), int(v))
        current = best.get(key)
        if current is None or weight < current[0]:
            best[key] = (float(weight), int(row))
    return best
