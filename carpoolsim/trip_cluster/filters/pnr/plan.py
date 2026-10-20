"""Lots where two travelers can meet before sharing a ride."""

from __future__ import annotations

import numpy as np


class PnrPlan:
    """Parking lots and the lot chosen for each ordered pair.

    ``px`` and ``py`` are lot coordinates in EPSG:3857 meters. ``r_p`` is the
    farthest a traveler may drive to a lot, in kilometers. ``lot[i, j]`` is
    the lot index used when ``i`` drives and ``j`` rides, or -1 when that
    pair has no lot. The access and Euclidean reroute filters fill ``lot``.
    Later filters read it and leave it unchanged.
    """

    def __init__(self, px: np.ndarray, py: np.ndarray, r_p: float) -> None:
        if r_p < 0:
            raise ValueError("r_p must be zero or positive")
        self.r_p = float(r_p)
        self.px = np.array(px, dtype=np.float64)
        self.py = np.array(py, dtype=np.float64)
        if self.px.ndim != 1 or self.py.ndim != 1:
            raise ValueError("px and py must be flat arrays")
        if self.px.shape != self.py.shape:
            raise ValueError("px and py must have the same length")
        if self.px.size == 0:
            raise ValueError("at least one lot is required")
        self.lot: np.ndarray | None = None

    def __len__(self) -> int:
        """Number of lots."""
        return int(self.px.shape[0])

    def store(self, matrix: np.ndarray, best: np.ndarray) -> None:
        """Remember ``best`` where ``matrix`` is on, and -1 everywhere else."""
        chosen = np.full(matrix.shape, -1, dtype=np.int64)
        np.copyto(chosen, best, where=matrix)
        np.fill_diagonal(chosen, -1)
        self.lot = chosen

    def require(self, n: int) -> np.ndarray:
        """Return the chosen lots for an ``n``-traveler cluster."""
        if self.lot is None or self.lot.shape != (n, n):
            raise RuntimeError("no park-and-ride lot has been chosen for this cluster")
        return self.lot
