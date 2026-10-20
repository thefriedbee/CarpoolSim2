"""Travelers still waiting to be seen, and travelers eligible now."""

from __future__ import annotations

import numpy as np


class Pool:
    """Split one day of travelers into those outside the horizon and those in it.

    ``waiting`` and ``active`` are sets of traveler ids. ``waiting`` holds
    ids whose departure is still beyond the current horizon. ``active`` holds
    ids inside the current window. A traveler leaves ``active`` when they are
    matched or when the clock passes their departure. ``departures`` maps
    each traveler id to a departure time in minutes after midnight.
    """

    def __init__(self, departures: np.ndarray, ids: np.ndarray) -> None:
        self.departures = {
            int(traveler): float(minute)
            for traveler, minute in zip(ids, departures, strict=True)
        }
        self.waiting = set(self.departures)
        self.active: set[int] = set()

    def admit(self, t: float, horizon: float) -> None:
        """Move waiting trips into the window once their departure is visible.

        A trip is visible when its departure is at most ``t + horizon``.
        """
        limit = t + horizon
        entered = [i for i in self.waiting if self.departures[i] <= limit]
        for traveler in entered:
            self.waiting.remove(traveler)
            self.active.add(traveler)

    def remove_departed(self, t: float) -> list[int]:
        """Drop active trips that departed before ``t`` and return their ids.

        Ids come back in ascending order.
        """
        left = sorted(i for i in self.active if self.departures[i] < t)
        for traveler in left:
            self.active.remove(traveler)
        return left

    def remove(self, traveler: int) -> None:
        """Drop one active traveler who has been given a partner."""
        self.active.remove(traveler)

    def ids_in_order(self) -> list[int]:
        """Active ids sorted by departure time, then by id."""
        return sorted(self.active, key=lambda traveler: (self.departures[traveler], traveler))
