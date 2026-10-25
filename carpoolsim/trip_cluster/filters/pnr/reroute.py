"""Keep pairs whose driver reroute through a lot stays within an inflation limit."""

from __future__ import annotations

import numpy as np

from carpoolsim.trip_cluster.cluster import TripCluster
from carpoolsim.trip_cluster.filters.meters import as_float32
from carpoolsim.trip_cluster.filters.meters import infinities
from carpoolsim.trip_cluster.filters.pnr.plan import PnrPlan


class FilterPnrRerouteEuclidean:
    """Turn off pairs whose driver would travel more than ``mu_1`` times the direct trip.

    ``matrix[i, j]`` means ``i`` drives and ``j`` rides. The driver's path is
    the Euclidean length from ``i``'s origin to a lot, then to ``j``'s
    destination, then to ``i``'s destination. Only lots within ``plan.r_p`` of
    both origins are considered. The lot with the smallest ratio is stored in
    ``plan.lot``. ``mu_1`` is the maximum ratio of that length to ``i``'s own
    origin-to-destination length. A driver with no direct length is turned
    off. A pair that is already off stays off.
    """

    def __init__(self, plan: PnrPlan, mu_1: float) -> None:
        if mu_1 < 0:
            raise ValueError("mu_1 must be zero or positive")
        self.plan = plan
        self.mu_1 = float(mu_1)

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        """AND the driver-reroute condition into ``matrix`` and record the lot."""
        ox = as_float32(cluster.ox)
        oy = as_float32(cluster.oy)
        dx = as_float32(cluster.dx)
        dy = as_float32(cluster.dy)
        px = as_float32(self.plan.px)
        py = as_float32(self.plan.py)
        origin_lot = np.hypot(ox[:, None] - px[None, :], oy[:, None] - py[None, :])
        lot_dest = np.hypot(px[:, None] - dx[None, :], py[:, None] - dy[None, :])
        dropoff = np.hypot(dx[None, :] - dx[:, None], dy[None, :] - dy[:, None])
        near = origin_lot <= self.plan.r_p * 1000.0
        reroute = origin_lot[:, None, :] + lot_dest.T[None, :, :] + dropoff[:, :, None]
        reroute = np.where(near[:, None, :] & near[None, :, :], reroute, np.float32(np.inf))
        direct = np.hypot(ox - dx, oy - dy)
        ratio = np.divide(
            reroute,
            direct[:, None, None],
            out=infinities(reroute.shape),
            where=direct[:, None, None] > 0,
        )
        matrix &= ratio.min(axis=2) <= self.mu_1
        self.plan.store(matrix, ratio.argmin(axis=2))
