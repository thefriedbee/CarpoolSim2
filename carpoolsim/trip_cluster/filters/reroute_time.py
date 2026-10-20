"""Keep pairs whose driver reroute stays within a network travel-time limit."""

from __future__ import annotations

import numpy as np
from pyproj import Transformer

from carpoolsim.network_prep.router import RoadRouter
from carpoolsim.trip_cluster.cluster import TripCluster

_TO_LONLAT = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


class FilterRerouteTravelTime:
    """Turn off pairs whose driver would take more than ``gamma_1`` times the solo trip.

    ``matrix[i, j]`` means ``i`` drives and ``j`` rides. Both trips are snapped
    to the drive network. The carpool time is the free-flow drive from ``i``'s
    origin to ``j``'s origin, then to ``j``'s destination, then to ``i``'s
    destination. ``gamma_1`` is the maximum ratio of that time to ``i``'s own
    origin-to-destination time. A driver with no solo path is turned off, as
    is a pair the network cannot connect. A pair that is already off stays off.
    """

    def __init__(self, router: RoadRouter, gamma_1: float) -> None:
        if gamma_1 < 0:
            raise ValueError("gamma_1 must be zero or positive")
        self.router = router
        self.gamma_1 = float(gamma_1)
        self._minutes: dict[tuple[int, int], float] = {}

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        """AND the network-reroute condition into ``matrix``."""
        origin = np.array([self._snap(x, y) for x, y in zip(cluster.ox, cluster.oy, strict=True)])
        dest = np.array([self._snap(x, y) for x, y in zip(cluster.dx, cluster.dy, strict=True)])
        solo = np.array([self._travel_minutes(int(origin[i]), int(dest[i])) for i in range(len(cluster))])
        ratio = np.full(matrix.shape, np.inf)
        for i in range(len(cluster)):
            if not solo[i] > 0:
                continue
            for j in range(len(cluster)):
                if i == j or not matrix[i, j]:
                    continue
                carpool = (
                    self._travel_minutes(int(origin[i]), int(origin[j]))
                    + self._travel_minutes(int(origin[j]), int(dest[j]))
                    + self._travel_minutes(int(dest[j]), int(dest[i]))
                )
                ratio[i, j] = carpool / solo[i]
        matrix &= ratio <= self.gamma_1

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
