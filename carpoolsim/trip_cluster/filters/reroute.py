"""Keep pairs whose driver reroute stays within an inflation limit."""

from __future__ import annotations

import numpy as np

from carpoolsim.trip_cluster.cluster import TripCluster


class FilterRerouteEuclideanDistance:
    """Turn off pairs whose driver would travel more than ``mu_1`` times the direct trip.

    ``matrix[i, j]`` means ``i`` drives and ``j`` rides. The driver's reroute
    is the Euclidean path from ``i``'s origin to ``j``'s origin, then to
    ``j``'s destination, then to ``i``'s destination. ``mu_1`` is the maximum
    ratio of that length to ``i``'s own origin-to-destination length. A driver
    with no direct length is turned off. A pair that is already off stays off.
    """

    def __init__(self, mu_1: float) -> None:
        if mu_1 < 0:
            raise ValueError("mu_1 must be zero or positive")
        self.mu_1 = float(mu_1)

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        """AND the driver-reroute condition into ``matrix``."""
        pickup = np.hypot(
            cluster.ox[:, None] - cluster.ox[None, :],
            cluster.oy[:, None] - cluster.oy[None, :],
        )
        passenger = np.hypot(cluster.ox - cluster.dx, cluster.oy - cluster.dy)
        dropoff = np.hypot(
            cluster.dx[None, :] - cluster.dx[:, None],
            cluster.dy[None, :] - cluster.dy[:, None],
        )
        direct = np.hypot(cluster.ox - cluster.dx, cluster.oy - cluster.dy)
        reroute = pickup + passenger[None, :] + dropoff
        ratio = np.divide(
            reroute,
            direct[:, None],
            out=np.full(reroute.shape, np.inf),
            where=direct[:, None] > 0,
        )
        matrix &= ratio <= self.mu_1
