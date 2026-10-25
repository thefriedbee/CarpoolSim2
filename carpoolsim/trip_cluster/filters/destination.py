"""Keep pairs whose destinations fall within a distance limit."""

from __future__ import annotations

import numpy as np

from carpoolsim.trip_cluster.cluster import TripCluster
from carpoolsim.trip_cluster.filters.meters import as_float32


class FilterDestinationEuclideanDistance:
    """Turn off pairs whose destinations are more than ``r2`` kilometers apart.

    Destinations are ``dx`` and ``dy`` in EPSG:3857 meters. The Euclidean
    distance is compared in kilometers. The check is symmetric, and a pair
    that is already off stays off.
    """

    def __init__(self, r2: float) -> None:
        if r2 < 0:
            raise ValueError("r2 must be zero or positive")
        self.r2 = float(r2)

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        """AND the destination-distance condition into ``matrix``."""
        east = as_float32(cluster.dx)
        north = as_float32(cluster.dy)
        distance = np.hypot(east[:, None] - east[None, :], north[:, None] - north[None, :])
        matrix &= distance <= self.r2 * 1000.0
