"""Split one pool into smaller clusters of similar trips, and stack them back."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from carpoolsim.trip_cluster.cluster import TripCluster

_MAX_STEPS = 32


def group_trips(cluster: TripCluster, max_size: int) -> list[TripCluster]:
    """Return disjoint clusters of at most ``max_size`` similar trips.

    Similarity is Euclidean distance between ``(ox, oy, dx, dy)`` in
    EPSG:3857 meters. Departure time is not part of the distance. The largest
    cluster is cut by 2-means until every cluster is within ``max_size``. A
    cut that cannot separate the points splits the rows in their current
    order. Each cluster keeps the departure order of the rows it takes, and
    keeps their traveler ids. Clusters come back in the order of their
    earliest row.
    """
    if isinstance(max_size, bool) or not isinstance(max_size, int):
        raise ValueError("max_size must be an integer")
    if max_size < 2:
        raise ValueError("max_size must be at least 2")
    count = len(cluster)
    if count == 0:
        return []
    points = np.column_stack((cluster.ox, cluster.oy, cluster.dx, cluster.dy))
    pending = [np.arange(count, dtype=np.int64)]
    finished: list[np.ndarray] = []
    while pending:
        rows = pending.pop()
        if int(rows.shape[0]) <= max_size:
            finished.append(rows)
            continue
        left, right = _bisect(points[rows])
        _require_split(rows.shape[0], left, right)
        pending.append(rows[np.sort(left)])
        pending.append(rows[np.sort(right)])
    finished.sort(key=lambda rows: int(rows[0]))
    return [_take(cluster, rows) for rows in finished]


def combine_trips(groups: Sequence[TripCluster]) -> TripCluster:
    """Stack ``groups`` in the order given.

    Row order inside each group is kept. Traveler ids must be unique across
    the groups. An empty sequence is an empty cluster.
    """
    if len(groups) == 0:
        empty = np.zeros(0, dtype=np.float64)
        return TripCluster(empty, empty, empty, empty, empty)
    return TripCluster(
        np.concatenate([group.ox for group in groups]),
        np.concatenate([group.oy for group in groups]),
        np.concatenate([group.dx for group in groups]),
        np.concatenate([group.dy for group in groups]),
        np.concatenate([group.t for group in groups]),
        np.concatenate([group.ids for group in groups]),
    )


def _take(cluster: TripCluster, rows: np.ndarray) -> TripCluster:
    """Copy the rows of ``cluster`` in the order ``rows`` gives."""
    return TripCluster(
        cluster.ox[rows],
        cluster.oy[rows],
        cluster.dx[rows],
        cluster.dy[rows],
        cluster.t[rows],
        cluster.ids[rows],
    )


def _bisect(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two non-empty row selections of ``points``.

    The seeds are the first row and the row farthest from it. Later steps
    assign every row to the nearer seed. A tie stays with the first seed.
    Identical rows are split in order instead.
    """
    count = int(points.shape[0])
    spread = np.sum((points - points[0]) ** 2, axis=1)
    far = int(np.argmax(spread))
    if spread[far] == 0.0:
        return _split_half(count)
    centers = np.stack([points[0].copy(), points[far].copy()])
    labels = np.zeros(count, dtype=np.int64)
    for _ in range(_MAX_STEPS):
        distance = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        updated = np.argmin(distance, axis=1).astype(np.int64)
        if np.array_equal(updated, labels):
            break
        labels = updated
        if int(labels.min()) == int(labels.max()):
            return _split_half(count)
        for side in (0, 1):
            centers[side] = points[labels == side].mean(axis=0)
    if int(labels.min()) == int(labels.max()):
        return _split_half(count)
    return np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)


def _split_half(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Split ``count`` rows into the earlier half and the later half."""
    mid = count // 2
    return np.arange(mid, dtype=np.int64), np.arange(mid, count, dtype=np.int64)


def _require_split(count: int, left: np.ndarray, right: np.ndarray) -> None:
    """Raise when a cut does not partition the rows into two smaller sides."""
    joined = np.concatenate([left, right])
    if left.size == 0 or right.size == 0 or np.unique(joined).size != count:
        raise RuntimeError("bisect did not split the cluster")
