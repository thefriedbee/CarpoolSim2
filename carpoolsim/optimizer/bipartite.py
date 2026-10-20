"""Maximum bipartite matching, then a greedy repair for role conflicts."""

from __future__ import annotations

import numpy as np
import rustworkx

from carpoolsim.optimizer.matrix import aligned_ids


class BipartiteAssigner:
    """Match driver seats to rider seats, then drop pairs that reuse a person.

    The driver seat and the rider seat of one traveler are different
    vertices, with an edge for every true cell of ``matrix``. A maximum
    bipartite matching can therefore give one person both seats, in two
    different cars. Those pairs are not a feasible carpool. The repair walks
    the matched cells in row-major order and keeps a pair only when both
    traveler ids are still free, recording taken ids in a set. A cell whose
    two ids are equal is not kept.

    The repair can drop a pair that an exact one-seat matching would have
    rearranged, so the result may use fewer cars than ``MatchingAssigner``.
    Pairs come back in the order the repair kept them.
    """

    def __call__(self, matrix: np.ndarray, ids: np.ndarray) -> list[tuple[int, int]]:
        """Return conflict-free ``(driver, rider)`` pairs of traveler ids."""
        labels = aligned_ids(matrix, ids)
        graph = rustworkx.PyGraph()
        driver_nodes = list(graph.add_nodes_from(range(len(labels))))
        rider_nodes = list(graph.add_nodes_from(range(len(labels))))
        rows, cols = np.nonzero(matrix)
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            driver = int(labels[row])
            rider = int(labels[col])
            if driver == rider:
                continue
            graph.add_edge(
                driver_nodes[row],
                rider_nodes[col],
                (int(row), int(col), driver, rider),
            )
        matched = [
            graph.get_edge_data(left, right)
            for left, right in rustworkx.max_weight_matching(graph, max_cardinality=True)
        ]
        matched.sort()
        used: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for _row, _col, driver, rider in matched:
            if driver in used or rider in used:
                continue
            used.add(driver)
            used.add(rider)
            pairs.append((driver, rider))
        return pairs
