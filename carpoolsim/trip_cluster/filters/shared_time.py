"""Keep pairs whose shared ride is a large enough share of the driver's tour."""

from __future__ import annotations

import numpy as np
from pyproj import Transformer

from carpoolsim.network_prep.router import RoadRouter
from carpoolsim.trip_cluster.cluster import TripCluster

_TO_LONLAT = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


class FilterSharedTravelTime:
    """Turn off pairs where the shared ride is less than ``gamma_2`` of the tour.

    ``matrix[i, j]`` means ``i`` drives and ``j`` rides. The driver's tour is
    the free-flow drive from ``i``'s origin to ``j``'s origin, then to ``j``'s
    destination, then to ``i``'s destination. The shared ride is the middle
    leg, while ``j`` is in the car. ``gamma_2`` is the minimum ratio of that
    leg to the whole tour, from 0 to 1. A tour with no travel time is turned
    off, as is a pair the network cannot connect. A pair that is already off
    stays off.
    """

    def __init__(self, router: RoadRouter, gamma_2: float) -> None:
        if not 0 <= gamma_2 <= 1:
            raise ValueError("gamma_2 must be between 0 and 1")
        self.router = router
        self.gamma_2 = float(gamma_2)
        self._minutes: dict[tuple[int, int], float] = {}

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        """AND the shared-ride condition into ``matrix``."""
        origin = np.array([self._snap(x, y) for x, y in zip(cluster.ox, cluster.oy, strict=True)])
        dest = np.array([self._snap(x, y) for x, y in zip(cluster.dx, cluster.dy, strict=True)])
        ratio = np.full(matrix.shape, np.nan)
        for i in range(len(cluster)):
            for j in range(len(cluster)):
                if i == j or not matrix[i, j]:
                    continue
                pickup = self._travel_minutes(int(origin[i]), int(origin[j]))
                shared = self._travel_minutes(int(origin[j]), int(dest[j]))
                dropoff = self._travel_minutes(int(dest[j]), int(dest[i]))
                total = pickup + shared + dropoff
                if not (np.isfinite(total) and total > 0):
                    continue
                ratio[i, j] = shared / total
        matrix &= ratio >= self.gamma_2

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
                minutes = self.router.route(origin, destination).travel_time
            except RuntimeError:
                minutes = float("inf")
            self._minutes[key] = minutes
        return minutes
