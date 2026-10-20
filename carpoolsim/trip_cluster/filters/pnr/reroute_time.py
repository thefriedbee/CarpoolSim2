"""Keep pairs whose driver reroute through a lot stays within a travel-time limit."""

from __future__ import annotations

import numpy as np

from carpoolsim.network_prep.router import RoadRouter
from carpoolsim.trip_cluster.cluster import TripCluster
from carpoolsim.trip_cluster.filters.pnr.plan import PnrPlan
from carpoolsim.trip_cluster.filters.pnr.routing import LotTravel


class FilterPnrRerouteTravelTime:
    """Turn off pairs whose driver would take more than ``gamma_1`` times the solo trip.

    ``matrix[i, j]`` means ``i`` drives and ``j`` rides. Trips and lots are
    snapped to the drive network. The carpool time is the free-flow drive
    from ``i``'s origin to ``plan.lot[i, j]``, then to ``j``'s destination,
    then to ``i``'s destination. ``gamma_1`` is the maximum ratio of that
    time to ``i``'s own origin-to-destination time. A driver with no solo
    path is turned off, as is a pair with no chosen lot or no connecting
    path. A pair that is already off stays off.
    """

    def __init__(self, plan: PnrPlan, router: RoadRouter, gamma_1: float) -> None:
        if gamma_1 < 0:
            raise ValueError("gamma_1 must be zero or positive")
        self.plan = plan
        self.gamma_1 = float(gamma_1)
        self._travel = LotTravel(router)

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        """AND the network-reroute condition into ``matrix``."""
        chosen = self.plan.require(len(cluster))
        origin = self._travel.nodes(cluster.ox, cluster.oy)
        dest = self._travel.nodes(cluster.dx, cluster.dy)
        park = self._travel.nodes(self.plan.px, self.plan.py)
        solo = np.array(
            [self._travel.minutes(int(origin[i]), int(dest[i])) for i in range(len(cluster))]
        )
        ratio = np.full(matrix.shape, np.inf)
        for i in range(len(cluster)):
            if not solo[i] > 0:
                continue
            for j in range(len(cluster)):
                if i == j or not matrix[i, j]:
                    continue
                lot = int(chosen[i, j])
                if lot < 0:
                    continue
                carpool = (
                    self._travel.minutes(int(origin[i]), int(park[lot]))
                    + self._travel.minutes(int(park[lot]), int(dest[j]))
                    + self._travel.minutes(int(dest[j]), int(dest[i]))
                )
                ratio[i, j] = carpool / solo[i]
        matrix &= ratio <= self.gamma_1
