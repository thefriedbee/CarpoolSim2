"""Split a drive network into parts along sparse geographic cuts.

Each cut keeps at least a fixed share of the nodes on both sides and crosses
as few streets as possible. A portal is a node with a neighbor in another part.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np

# East, north, and the two diagonals. Coordinates are EPSG:3857 meters.
_AXES = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [np.sqrt(0.5), np.sqrt(0.5)],
        [np.sqrt(0.5), -np.sqrt(0.5)],
    ]
)


class NetworkPartition:
    """Parts of a drive network and the portals on their boundaries.

    Node positions are row numbers in the node table. ``src`` and ``dst`` are
    the endpoint positions of each edge row. A portal has a neighbor in another
    part.
    """

    def __init__(
        self,
        xy: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
        parts: list[np.ndarray],
        cell_of: np.ndarray,
        portal_ids: np.ndarray,
        is_portal: np.ndarray,
        n_links: int,
        n_boundary_links: int,
    ) -> None:
        self.xy = xy
        self.src = src
        self.dst = dst
        self.parts = parts
        self.cell_of = cell_of
        self.portal_ids = portal_ids
        self.is_portal = is_portal
        self.n_links = n_links
        self.n_boundary_links = n_boundary_links

    @property
    def n_parts(self) -> int:
        """Number of parts."""
        return len(self.parts)

    @property
    def n_portals(self) -> int:
        """Number of portal nodes."""
        return int(self.portal_ids.size)

    def part_sizes(self) -> np.ndarray:
        """Return the number of nodes in each part."""
        return np.array([len(part) for part in self.parts])

    def portal_counts(self) -> np.ndarray:
        """Return the number of portals in each part."""
        return np.array([int(self.is_portal[members].sum()) for members in self.parts])


def endpoints(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return node coordinates and the row position of each edge endpoint.

    ``xy`` has ``x`` and ``y`` in EPSG:3857. ``src[i]`` and ``dst[i]`` are the
    node-table rows joined by edge row ``i``.
    """
    index_of = {int(osmid): i for i, osmid in enumerate(nodes.index.to_numpy())}
    xy = np.column_stack([nodes["x"].to_numpy(), nodes["y"].to_numpy()])
    src = np.fromiter(
        (index_of[int(u)] for u, _, _ in edges.index),
        dtype=np.int32,
        count=len(edges),
    )
    dst = np.fromiter(
        (index_of[int(v)] for _, v, _ in edges.index),
        dtype=np.int32,
        count=len(edges),
    )
    return xy, src, dst


def partition_network(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    *,
    max_nodes: int = 400,
    min_share: float = 0.30,
) -> NetworkPartition:
    """Split ``nodes`` and ``edges`` into parts of at most ``max_nodes`` nodes.

    Each cut is the sparsest one among four directions. Each side keeps at
    least ``min_share`` of the nodes, so ``min_share`` must be below one half.
    """
    if max_nodes < 2:
        raise ValueError("max_nodes must be at least 2")
    if not 0 < min_share < 0.5:
        raise ValueError("min_share must be between 0 and 0.5")

    xy, src, dst = endpoints(nodes, edges)
    left = np.minimum(src, dst)
    right = np.maximum(src, dst)
    keep = left != right
    links = np.unique(np.stack([left[keep], right[keep]], axis=1), axis=0)
    parts = _split_parts(
        xy,
        np.arange(len(xy), dtype=np.int32),
        links[:, 0],
        links[:, 1],
        max_nodes,
        min_share,
    )

    cell_of = np.empty(len(xy), dtype=np.int32)
    for cell_id, members in enumerate(parts):
        cell_of[members] = cell_id
    cross = cell_of[links[:, 0]] != cell_of[links[:, 1]]
    portal_ids = np.unique(np.concatenate([links[cross, 0], links[cross, 1]]))
    is_portal = np.zeros(len(xy), dtype=bool)
    is_portal[portal_ids] = True
    return NetworkPartition(
        xy=xy,
        src=src,
        dst=dst,
        parts=parts,
        cell_of=cell_of,
        portal_ids=portal_ids,
        is_portal=is_portal,
        n_links=int(len(links)),
        n_boundary_links=int(cross.sum()),
    )


def _split_parts(
    xy: np.ndarray,
    members: np.ndarray,
    eu: np.ndarray,
    ev: np.ndarray,
    max_nodes: int,
    min_share: float,
) -> list[np.ndarray]:
    """Return leaf groups of at most ``max_nodes`` nodes."""
    count = len(members)
    if count <= max_nodes:
        return [members]

    best = None
    coords = xy[members]
    n_nodes = len(xy)
    for axis in _AXES:
        order = np.argsort(coords @ axis, kind="mergesort")
        ordered = members[order]
        rank = np.full(n_nodes, -1, dtype=np.int32)
        rank[ordered] = np.arange(count, dtype=np.int32)
        ru = rank[eu]
        rv = rank[ev]
        inside = (ru >= 0) & (rv >= 0)
        lo = np.minimum(ru[inside], rv[inside])
        hi = np.maximum(ru[inside], rv[inside])
        span = lo < hi
        lo = lo[span]
        hi = hi[span]
        # cut[s] counts links with one end in ordered[:s] and the other outside it.
        diff = np.zeros(count + 1, dtype=np.int32)
        np.add.at(diff, lo + 1, 1)
        np.add.at(diff, hi + 1, -1)
        cut = np.cumsum(diff[:-1])
        s_min = int(np.ceil(min_share * count))
        s_max = int(np.floor((1.0 - min_share) * count))
        window = cut[s_min : s_max + 1]
        choice = int(np.argmin(window))
        size = s_min + choice
        crossing = int(window[choice])
        if best is None or crossing < best[0]:
            best = (crossing, ordered[:size], ordered[size:])

    _, left_nodes, right_nodes = best
    on_left = np.zeros(n_nodes, dtype=bool)
    on_left[left_nodes] = True
    u_left = on_left[eu]
    v_left = on_left[ev]
    stay_left = u_left & v_left
    stay_right = ~u_left & ~v_left
    return _split_parts(
        xy, left_nodes, eu[stay_left], ev[stay_left], max_nodes, min_share
    ) + _split_parts(xy, right_nodes, eu[stay_right], ev[stay_right], max_nodes, min_share)
