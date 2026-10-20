"""An ordered group of trips for later matrix work."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class TripCluster:
    """Trips in a fixed order.

    ``ox``, ``oy``, ``dx``, ``dy``, ``t``, and ``ids`` are flat arrays of the
    same length. ``ox`` and ``oy`` are the origin in EPSG:3857 meters, ``dx``
    and ``dy`` are the destination in the same coordinates, and ``t`` is the
    departure time in minutes after midnight. ``ids[i]`` is the traveler in
    row ``i``. Ids are unique integers, so a subset of the cluster can still
    name the trips it came from. When ``ids`` is omitted, rows are numbered
    from zero in order.
    """

    def __init__(
        self,
        ox: np.ndarray,
        oy: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        t: np.ndarray,
        ids: np.ndarray | None = None,
    ) -> None:
        self.ox = np.array(ox, dtype=np.float64)
        self.oy = np.array(oy, dtype=np.float64)
        self.dx = np.array(dx, dtype=np.float64)
        self.dy = np.array(dy, dtype=np.float64)
        self.t = np.array(t, dtype=np.float64)
        arrays = (self.ox, self.oy, self.dx, self.dy, self.t)
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("ox, oy, dx, dy, and t must be flat arrays")
        if any(array.shape != self.ox.shape for array in arrays):
            raise ValueError("ox, oy, dx, dy, and t must have the same length")
        if ids is None:
            self.ids = np.arange(self.ox.shape[0], dtype=np.int64)
        else:
            self.ids = np.array(ids, dtype=np.int64)
            if self.ids.ndim != 1 or self.ids.shape != self.ox.shape:
                raise ValueError("ids must be a flat array of the same length")
            if len(set(self.ids.tolist())) != self.ids.shape[0]:
                raise ValueError("ids must be unique")

    def __len__(self) -> int:
        """Number of travelers."""
        return int(self.ox.shape[0])

    @classmethod
    def from_csv(cls, path: str | Path) -> TripCluster:
        """Load trips from a CSV of ``origin_x/y``, ``dest_x/y``, and ``departure_time``.

        ``departure_time`` is ``HH:MM``. Rows stay in file order. A ``trip_id``
        column becomes ``ids``. Without that column, rows are numbered from
        zero in file order.
        """
        ox: list[float] = []
        oy: list[float] = []
        dx: list[float] = []
        dy: list[float] = []
        t: list[float] = []
        ids: list[int] = []
        with Path(path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            has_id = "trip_id" in (reader.fieldnames or [])
            for row_index, row in enumerate(reader):
                ox.append(float(row["origin_x"]))
                oy.append(float(row["origin_y"]))
                dx.append(float(row["dest_x"]))
                dy.append(float(row["dest_y"]))
                t.append(float(minutes_of_day(row["departure_time"])))
                ids.append(int(row["trip_id"]) if has_id else row_index)
        return cls(ox, oy, dx, dy, t, ids)


def minutes_of_day(clock: str) -> int:
    """Minutes after midnight for a clock like ``07:38``."""
    hour_text, minute_text = clock.strip().split(":")
    return int(hour_text) * 60 + int(minute_text)
