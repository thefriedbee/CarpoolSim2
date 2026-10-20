"""Keep pairs whose departure times fall within a limit."""

from __future__ import annotations

import numpy as np

from carpoolsim.trip_cluster.cluster import TripCluster


class FilterDepartureTime:
    """Turn off pairs whose departure times differ by more than ``delta1``.

    ``delta1`` is in minutes, the same units as ``TripCluster.t``. The check
    is symmetric: pair ``(i, j)`` stays only when ``abs(t[i] - t[j]) <= delta1``.
    A pair that is already off stays off.
    """

    def __init__(self, delta1: float) -> None:
        if delta1 < 0:
            raise ValueError("delta1 must be zero or positive")
        self.delta1 = float(delta1)

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        """AND the departure-time condition into ``matrix``."""
        within = np.abs(cluster.t[:, None] - cluster.t[None, :]) <= self.delta1
        matrix &= within
