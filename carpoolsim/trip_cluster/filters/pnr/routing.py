"""Snap park-and-ride points and cache free-flow travel times."""

from __future__ import annotations

import numpy as np
from pyproj import Transformer

from carpoolsim.network_prep.router import RoadRouter

_TO_LONLAT = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


class LotTravel:
    """Free-flow minutes between snapped trip ends and lots."""

    def __init__(self, router: RoadRouter) -> None:
        self.router = router
        self._minutes: dict[tuple[int, int], float] = {}

    def nodes(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the drive-network node nearest to each EPSG:3857 point."""
        return np.array([self._snap(float(a), float(b)) for a, b in zip(x, y, strict=True)])

    def minutes(self, origin: int, destination: int) -> float:
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

    def _snap(self, x: float, y: float) -> int:
        lon, lat = _TO_LONLAT.transform(x, y)
        return self.router.nearest_node(lon, lat)
