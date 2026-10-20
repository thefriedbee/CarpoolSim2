"""Take feasible pairs in scan order, skipping any traveler already used."""

from __future__ import annotations

import numpy as np

from carpoolsim.optimizer.matrix import aligned_ids


class GreedyAssigner:
    """Take feasible pairs in row-major order, using each traveler once.

    Scans ``matrix[i, j]`` with row ``i`` as the driver and row ``j`` as the
    rider. ``ids[i]`` is the traveler that row stands for. The first true
    cell whose two ids are both still free becomes a pair. Later cells that
    reuse either id are skipped. Taken ids are kept in a set, so the same
    traveler cannot take a second seat when this matrix is a reordered
    subset of a larger day. A cell whose two ids are equal is skipped.
    """

    def __call__(self, matrix: np.ndarray, ids: np.ndarray) -> list[tuple[int, int]]:
        """Return disjoint ``(driver, rider)`` pairs of traveler ids."""
        labels = aligned_ids(matrix, ids)
        used: set[int] = set()
        pairs: list[tuple[int, int]] = []
        drivers, riders = np.nonzero(matrix)
        for row, col in zip(drivers, riders, strict=True):
            driver = int(labels[row])
            rider = int(labels[col])
            if driver in used or rider in used or driver == rider:
                continue
            used.add(driver)
            used.add(rider)
            pairs.append((driver, rider))
        return pairs
