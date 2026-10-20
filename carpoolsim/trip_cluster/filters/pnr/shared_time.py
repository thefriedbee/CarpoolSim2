"""Keep pairs whose shared ride from the lot is a large enough share of the tour."""

from __future__ import annotations

import numpy as np

from carpoolsim.network_prep.router import RoadRouter
from carpoolsim.trip_cluster.cluster import TripCluster
from carpoolsim.trip_cluster.filters.pnr.plan import PnrPlan
from carpoolsim.trip_cluster.filters.pnr.routing import LotTravel


class FilterPnrSharedTravelTime:
    """Turn off pairs where the shared ride is less than ``gamma_2`` of the tour.

    ``matrix[i, j]`` means ``i`` drives and ``j`` rides. The driver's tour is
    the free-flow drive from ``i``'s origin to ``plan.lot[i, j]``, then to
    ``j``'s destination, then to ``i``'s destination. The shared ride is the
    middle leg, from the lot until ``j`` gets out. ``gamma_2`` is the minimum
    ratio of that leg to the whole tour, from 0 to 1. A tour with no travel
    time is turned off, as is a pair with no chosen lot or no connecting
    path. A pair that is already off stays off.
    """

    def __init__(self, plan: PnrPlan, router: RoadRouter, gamma_2: float) -> None:
        if not 0 <= gamma_2 <= 1:
            raise ValueError("gamma_2 must be between 0 and 1")
        self.plan = plan
        self.gamma_2 = float(gamma_2)
        self._travel = LotTravel(router)

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        """AND the shared-ride condition into ``matrix``."""
        chosen = self.plan.require(len(cluster))
        origin = self._travel.nodes(cluster.ox, cluster.oy)
        dest = self._travel.nodes(cluster.dx, cluster.dy)
        park = self._travel.nodes(self.plan.px, self.plan.py)
        ratio = np.full(matrix.shape, np.nan)
        for i in range(len(cluster)):
            for j in range(len(cluster)):
                if i == j or not matrix[i, j]:
                    continue
                lot = int(chosen[i, j])
                if lot < 0:
                    continue
                pickup = self._travel.minutes(int(origin[i]), int(park[lot]))
                shared = self._travel.minutes(int(park[lot]), int(dest[j]))
                dropoff = self._travel.minutes(int(dest[j]), int(dest[i]))
                total = pickup + shared + dropoff
                if not (np.isfinite(total) and total > 0):
                    continue
                ratio[i, j] = shared / total
        matrix &= ratio >= self.gamma_2
