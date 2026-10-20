"""Keep pairs whose final drop-off leg stays within a share of the solo trip."""

from __future__ import annotations

import numpy as np

from carpoolsim.trip_cluster.cluster import TripCluster


class FilterDropoffEuclideanDistance:
    """Turn off pairs whose last leg is more than ``mu_2`` of the driver's solo trip.

    ``matrix[i, j]`` means ``i`` drives and ``j`` rides. The last leg is the
    Euclidean distance from ``j``'s destination to ``i``'s destination, after
    ``i`` has dropped ``j`` off. ``mu_2`` is the maximum ratio of that length
    to ``i``'s own origin-to-destination length. A driver with no direct
    length is turned off. A pair that is already off stays off.
    """

    def __init__(self, mu_2: float) -> None:
        if mu_2 < 0:
            raise ValueError("mu_2 must be zero or positive")
        self.mu_2 = float(mu_2)

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        """AND the drop-off condition into ``matrix``."""
        last_leg = np.hypot(
            cluster.dx[None, :] - cluster.dx[:, None],
            cluster.dy[None, :] - cluster.dy[:, None],
        )
        direct = np.hypot(cluster.ox - cluster.dx, cluster.oy - cluster.dy)
        ratio = np.divide(
            last_leg,
            direct[:, None],
            out=np.full(last_leg.shape, np.inf),
            where=direct[:, None] > 0,
        )
        matrix &= ratio <= self.mu_2
