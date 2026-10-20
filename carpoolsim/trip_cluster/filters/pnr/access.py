"""Keep pairs that can both reach one parking lot."""

from __future__ import annotations

import numpy as np

from carpoolsim.trip_cluster.cluster import TripCluster
from carpoolsim.trip_cluster.filters.pnr.plan import PnrPlan


class FilterPnrAccess:
    """Turn off pairs that cannot both reach a lot within ``r_p``.

    ``r_p`` is ``plan.r_p``, in kilometers. A pair stays on when some lot is
    within that distance of both origins. The chosen lot minimizes the sum of
    those two distances, and the lowest lot index wins a tie. ``plan.lot``
    stores that index, or -1 when the pair is off. The check is symmetric. A
    pair that is already off stays off.
    """

    def __init__(self, plan: PnrPlan) -> None:
        self.plan = plan

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        """AND the lot-access condition into ``matrix`` and record the lot."""
        origin_lot = np.hypot(
            cluster.ox[:, None] - self.plan.px[None, :],
            cluster.oy[:, None] - self.plan.py[None, :],
        )
        near = origin_lot <= self.plan.r_p * 1000.0
        cost = np.where(
            near[:, None, :] & near[None, :, :],
            origin_lot[:, None, :] + origin_lot[None, :, :],
            np.inf,
        )
        matrix &= np.isfinite(cost.min(axis=2))
        self.plan.store(matrix, cost.argmin(axis=2))
