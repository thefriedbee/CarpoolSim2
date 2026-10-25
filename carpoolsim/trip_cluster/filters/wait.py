"""Keep pairs whose passenger waits no longer than a limit for the driver."""

from __future__ import annotations

import numpy as np
from pyproj import Transformer

from carpoolsim.network_prep.router import RoadRouter
from carpoolsim.trip_cluster.cluster import TripCluster

_TO_LONLAT = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


class FilterPassengerWaitTime:
    """Turn off pairs where passenger ``j`` would wait more than ``delta2`` minutes.

    ``matrix[i, j]`` means ``i`` drives and ``j`` rides. The driver leaves at
    ``t[i]`` and reaches ``j``'s origin after the free-flow drive from ``i``.
    The wait is that arrival time minus ``t[j]``. ``delta2`` is the maximum
    wait, in minutes. A negative wait means the driver arrives before ``j``'s
    stated departure, so ``j`` leaves early and the driver does not wait. A
    pair the network cannot connect is turned off. A pair that is already off
    stays off.
    """

    def __init__(self, router: RoadRouter, delta2: float) -> None:
        if delta2 < 0:
            raise ValueError("delta2 must be zero or positive")
        self.router = router
        self.delta2 = float(delta2)
        self._minutes: dict[tuple[int, int], float] = {}

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        """AND the passenger-wait condition into ``matrix``."""
        origin = np.array([self._snap(x, y) for x, y in zip(cluster.ox, cluster.oy, strict=True)])
        wait = np.full(matrix.shape, np.inf)
        for i in range(len(cluster)):
            for j in range(len(cluster)):
                if i == j or not matrix[i, j]:
                    continue
                drive = self._travel_minutes(int(origin[i]), int(origin[j]))
                if not np.isfinite(drive):
                    continue
                wait[i, j] = cluster.t[i] + drive - cluster.t[j]
        matrix &= wait <= self.delta2

    def _snap(self, x: float, y: float) -> int:
        """Return the drive-network node nearest to an EPSG:3857 point."""
        lon, lat = _TO_LONLAT.transform(x, y)
        return self.router.nearest_node(lon, lat)

    def _travel_minutes(self, origin: int, destination: int) -> float:
        """Free-flow minutes from ``origin`` to ``destination``, or infinity if none."""
        key = (origin, destination)
        minutes = self._minutes.get(key)
        if minutes is None:
            try:
                minutes = self.router.travel_minutes(origin, destination)
            except RuntimeError:
                minutes = float("inf")
            self._minutes[key] = minutes
        return minutes
