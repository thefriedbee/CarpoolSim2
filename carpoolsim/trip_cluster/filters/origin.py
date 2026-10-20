"""Keep pairs whose origins fall within a distance limit."""

from __future__ import annotations

import numpy as np

from carpoolsim.trip_cluster.cluster import TripCluster


class FilterOriginEuclideanDistance:
    """Turn off pairs whose origins are more than ``r1`` kilometers apart.

    Origins are ``ox`` and ``oy`` in EPSG:3857 meters. The Euclidean distance
    is compared in kilometers. The check is symmetric, and a pair that is
    already off stays off.
    """

    def __init__(self, r1: float) -> None:
        if r1 < 0:
            raise ValueError("r1 must be zero or positive")
        self.r1 = float(r1)

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        """AND the origin-distance condition into ``matrix``."""
        east = cluster.ox[:, None] - cluster.ox[None, :]
        north = cluster.oy[:, None] - cluster.oy[None, :]
        matrix &= np.hypot(east, north) <= self.r1 * 1000.0
