"""Maximum-cardinality matching with one seat per traveler."""

from __future__ import annotations

import numpy as np
import rustworkx

from carpoolsim.optimizer.matrix import aligned_ids


class MatchingAssigner:
    """Pick the largest set of pairs that gives every traveler one seat.

    Each traveler is one vertex. An undirected edge joins two travelers when
    at least one direction of ``matrix`` is true. A maximum-cardinality
    matching then selects edges that share no vertex, so nobody drives and
    rides, and nobody is in two cars. This is exact for pairs: it can keep
    two disjoint pairs that a row-major scan would block with an earlier cell.

    When both directions are feasible, the traveler with the smaller id
    drives. Pairs come back sorted by ``(driver, rider)``.
    """

    def __call__(self, matrix: np.ndarray, ids: np.ndarray) -> list[tuple[int, int]]:
        """Return a largest set of disjoint ``(driver, rider)`` traveler ids."""
        labels = aligned_ids(matrix, ids)
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(range(len(labels)))
        for row in range(len(labels)):
            for col in range(row + 1, len(labels)):
                pair = _directed_pair(matrix, labels, row, col)
                if pair is None:
                    continue
                graph.add_edge(row, col, pair)
        matched = [
            graph.get_edge_data(left, right)
            for left, right in rustworkx.max_weight_matching(graph, max_cardinality=True)
        ]
        return sorted(matched)


def _directed_pair(
    matrix: np.ndarray,
    labels: np.ndarray,
    row: int,
    col: int,
) -> tuple[int, int] | None:
    """Return who drives when either direction between ``row`` and ``col`` works.

    If both directions are feasible, the smaller traveler id drives. If
    neither direction is feasible, return nothing.
    """
    forward = bool(matrix[row, col])
    backward = bool(matrix[col, row])
    if forward and not backward:
        return int(labels[row]), int(labels[col])
    if backward and not forward:
        return int(labels[col]), int(labels[row])
    if not forward:
        return None
    if int(labels[row]) <= int(labels[col]):
        return int(labels[row]), int(labels[col])
    return int(labels[col]), int(labels[row])
