"""Keep pairs whose passenger can reach the chosen lot without too long a drive."""

from __future__ import annotations

import numpy as np

from carpoolsim.trip_cluster.cluster import TripCluster
from carpoolsim.trip_cluster.filters.meters import as_float32
from carpoolsim.trip_cluster.filters.meters import infinities
from carpoolsim.trip_cluster.filters.pnr.plan import PnrPlan


class FilterPnrPassengerAccess:
    """Turn off pairs whose passenger drives more than ``mu_3`` of their solo trip.

    ``matrix[i, j]`` means ``i`` drives and ``j`` rides. The access is the
    Euclidean distance from ``j``'s origin to ``plan.lot[i, j]``. ``mu_3`` is
    the maximum ratio of that length to ``j``'s own origin-to-destination
    length. A pair with no chosen lot is turned off, as is a passenger with
    no direct length. A pair that is already off stays off.
    """

    def __init__(self, plan: PnrPlan, mu_3: float) -> None:
        if mu_3 < 0:
            raise ValueError("mu_3 must be zero or positive")
        self.plan = plan
        self.mu_3 = float(mu_3)

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        """AND the passenger-access condition into ``matrix``."""
        chosen = self.plan.require(len(cluster))
        parked = np.clip(chosen, 0, None)
        ox = as_float32(cluster.ox)
        oy = as_float32(cluster.oy)
        dx = as_float32(cluster.dx)
        dy = as_float32(cluster.dy)
        px = as_float32(self.plan.px)
        py = as_float32(self.plan.py)
        access = np.hypot(ox[None, :] - px[parked], oy[None, :] - py[parked])
        solo = np.hypot(ox - dx, oy - dy)
        ratio = np.divide(
            access,
            solo[None, :],
            out=infinities(access.shape),
            where=solo[None, :] > 0,
        )
        ratio = np.where(chosen >= 0, ratio, np.float32(np.inf))
        matrix &= ratio <= self.mu_3
