"""Keep pairs whose earlier arrival waits no longer than a limit at the lot."""

from __future__ import annotations

import numpy as np

from carpoolsim.network_prep.router import RoadRouter
from carpoolsim.trip_cluster.cluster import TripCluster
from carpoolsim.trip_cluster.filters.pnr.plan import PnrPlan
from carpoolsim.trip_cluster.filters.pnr.routing import LotTravel


class FilterPnrWait:
    """Turn off pairs who would wait more than ``delta2`` minutes at the lot.

    ``matrix[i, j]`` means ``i`` drives and ``j`` rides. Each traveler leaves
    at their own departure time and drives free-flow to ``plan.lot[i, j]``.
    The wait is the absolute difference of those arrival times, so the person
    who reaches the lot first waits for the other. ``delta2`` is the maximum
    wait, in minutes. A pair with no chosen lot is turned off, as is a pair
    the network cannot connect. A pair that is already off stays off.
    """

    def __init__(self, plan: PnrPlan, router: RoadRouter, delta2: float) -> None:
        if delta2 < 0:
            raise ValueError("delta2 must be zero or positive")
        self.plan = plan
        self.delta2 = float(delta2)
        self._travel = LotTravel(router)

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        """AND the lot-wait condition into ``matrix``."""
        chosen = self.plan.require(len(cluster))
        origin = self._travel.nodes(cluster.ox, cluster.oy)
        park = self._travel.nodes(self.plan.px, self.plan.py)
        wait = np.full(matrix.shape, np.inf)
        for i in range(len(cluster)):
            for j in range(len(cluster)):
                if i == j or not matrix[i, j]:
                    continue
                lot = int(chosen[i, j])
                if lot < 0:
                    continue
                drive_i = self._travel.minutes(int(origin[i]), int(park[lot]))
                drive_j = self._travel.minutes(int(origin[j]), int(park[lot]))
                if not (np.isfinite(drive_i) and np.isfinite(drive_j)):
                    continue
                wait[i, j] = abs(cluster.t[i] + drive_i - (cluster.t[j] + drive_j))
        matrix &= wait <= self.delta2
