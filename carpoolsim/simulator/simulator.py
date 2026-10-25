"""Advance a clock and match the trips visible at each step."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import numpy as np

from carpoolsim.optimizer import Assigner
from carpoolsim.simulator.pool import Pool
from carpoolsim.trip_cluster.cluster import TripCluster
from carpoolsim.trip_cluster.grouping import group_trips
from carpoolsim.trip_cluster.share import Filter
from carpoolsim.trip_cluster.share import Shareability


# Most travelers kept in one spatial group. None leaves the pool as one group.
DEFAULT_MAX_SIZE = 100


class Match:
    """One committed carpool.

    ``driver`` and ``rider`` are traveler ids from the full-day cluster. ``t``
    is the clock, in minutes after midnight, when the pair was committed.
    ``mode`` is the pipeline that produced the pair. It is empty when the
    simulator was given one unnamed ``filters`` sequence.
    """

    def __init__(self, driver: int, rider: int, t: float, mode: str = "") -> None:
        self.driver = driver
        self.rider = rider
        self.t = float(t)
        self.mode = mode


class Solo:
    """One traveler who left without a partner.

    ``traveler`` is an id from the full-day cluster. ``t`` is the clock,
    in minutes after midnight, when that departure had already passed.
    """

    def __init__(self, traveler: int, t: float) -> None:
        self.traveler = traveler
        self.t = float(t)


class Simulator:
    """Match trips inside a moving departure window.

    The clock starts at the earliest departure minus ``horizon`` and advances
    by ``dt``. Both are in minutes, the same units as ``TripCluster.t``. At
    clock ``t`` the pool holds every still-unmatched traveler whose departure
    lies in ``[t, t + horizon]``. Modes then run in the order they were given.
    Each mode copies the travelers who are still unmatched into a cluster, in
    departure order. The cluster is cut into groups of at most ``max_size``
    similar trips, and each group is filtered and assigned on its own. The
    pairs from those groups are then committed together. The default group
    holds 100 travelers. Pass ``None`` and the cluster stays one group. The
    copy keeps their traveler ids. Both members of a committed pair leave
    the pool before the next mode runs. A later mode can match only the
    travelers the earlier modes left.
    A traveler who is still unmatched after every mode, and whose departure
    the clock has passed, is recorded as solo.

    ``modes`` is a sequence of ``(name, filters)`` pairs, such as direct
    carpool followed by park-and-ride. Each name is used once. ``filters``
    is a single unnamed mode for a run that has only one pipeline.

    ``commit`` is ``"window"`` or ``"step"``. ``"window"`` keeps every pair
    the assigner returns. ``"step"`` keeps a pair only when one member departs
    in ``[t, t + dt)``. A pair that is not kept stays in the pool, so a later
    mode in this tick, or a later tick, can still match those travelers.

    When ``et`` is set, a traveler in the pool may drive at clock ``t`` only
    if that departure lies in ``[t, t + et]``. A later departure in the pool
    can still ride. The simulator turns those driver rows off before the
    mode's filters run. When ``et`` is omitted, anyone in the pool may drive.
    A provided ``et`` is at least ``dt`` and at most ``horizon``.
    """

    def __init__(
        self,
        cluster: TripCluster,
        assigner: Assigner,
        filters: Sequence[Filter] = (),
        horizon: float = 15,
        dt: float = 5,
        et: float | None = None,
        commit: str = "window",
        modes: Sequence[tuple[str, Sequence[Filter]]] | None = None,
        max_size: int | None = DEFAULT_MAX_SIZE,
    ) -> None:
        self.cluster = cluster
        self.assigner = assigner
        self.modes = _mode_list(filters, modes)
        self.horizon = float(horizon)
        self.dt = float(dt)
        self.et = None if et is None else float(et)
        self.commit = commit
        self.max_size = None if max_size is None else _count(max_size, "max_size")
        self.matches: list[Match] = []
        self.solos: list[Solo] = []
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.horizon < self.dt:
            raise ValueError("horizon must be at least dt")
        if self.et is not None and self.et < self.dt:
            raise ValueError("et must be at least dt")
        if self.et is not None and self.et > self.horizon:
            raise ValueError("et must be at most horizon")
        if commit not in {"window", "step"}:
            raise ValueError("commit must be 'window' or 'step'")
        if self.max_size is not None and self.max_size < 2:
            raise ValueError("max_size must be at least 2")

    def run(self) -> None:
        """Step the clock until every traveler is matched or has departed."""
        self.matches = []
        self.solos = []
        if len(self.cluster) == 0:
            return
        pool = Pool(self.cluster.t, self.cluster.ids)
        t = float(np.min(self.cluster.t) - self.horizon)
        while True:
            pool.admit(t, self.horizon)
            for traveler in pool.remove_departed(t):
                self.solos.append(Solo(traveler, t))
            if not pool.waiting and not pool.active:
                break
            if pool.active:
                self._assign(pool, t)
            t += self.dt

    def _assign(self, pool: Pool, t: float) -> None:
        """Run each mode on the travelers the earlier modes left unmatched."""
        for name, filters in self.modes:
            if not pool.active:
                return
            active = _subset(self.cluster, pool.ids_in_order())
            groups = [active] if self.max_size is None else group_trips(active, self.max_size)
            pairs: list[tuple[int, int]] = []
            for group in groups:
                pairs.extend(_pairs(group, filters, self.assigner, t, self.et))
            _reject_repeated_travelers(pairs)
            for driver, rider in pairs:
                if self.commit == "step" and not _expires_this_step(
                    pool.departures, driver, rider, t, self.dt
                ):
                    continue
                self.matches.append(Match(driver, rider, t, name))
                pool.remove(driver)
                pool.remove(rider)


def _mode_list(
    filters: Sequence[Filter],
    modes: Sequence[tuple[str, Sequence[Filter]]] | None,
) -> tuple[tuple[str, tuple[Filter, ...]], ...]:
    """Return the ordered pipelines, or one unnamed pipeline from ``filters``."""
    if modes is None:
        return (("", tuple(filters)),)
    if len(tuple(filters)) > 0:
        raise ValueError("pass filters or modes, not both")
    named: list[tuple[str, tuple[Filter, ...]]] = []
    seen: set[str] = set()
    for name, mode_filters in modes:
        if not isinstance(name, str) or name == "":
            raise ValueError("mode name must be a non-empty string")
        if name in seen:
            raise ValueError("mode names must be unique")
        seen.add(name)
        named.append((name, tuple(mode_filters)))
    if not named:
        raise ValueError("at least one mode is required")
    return tuple(named)


def _pairs(
    cluster: TripCluster,
    filters: Sequence[Filter],
    assigner: Assigner,
    t: float,
    et: float | None,
) -> list[tuple[int, int]]:
    """Filter one group and return the assigner's pairs of traveler ids."""
    share = Shareability(cluster)
    if et is not None:
        _limit_drivers(share.cluster, share.matrix, t, et)
    share.apply(*filters)
    known = {int(traveler) for traveler in share.cluster.ids}
    return [
        _checked_pair(pair, known) for pair in assigner(share.matrix, share.cluster.ids)
    ]


def _count(value: object, label: str) -> int:
    """Return ``value`` as an integer, rejecting booleans and fractional numbers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _subset(cluster: TripCluster, ids: list[int]) -> TripCluster:
    """Copy the travelers in ``ids``, in that order, keeping those ids.

    Row ``i`` of the copy is traveler ``ids[i]``. The copy keeps those ids,
    so a matrix built from it still names the trips from the full day.
    """
    position = {int(traveler): index for index, traveler in enumerate(cluster.ids)}
    index = np.array([position[traveler] for traveler in ids], dtype=int)
    return TripCluster(
        cluster.ox[index],
        cluster.oy[index],
        cluster.dx[index],
        cluster.dy[index],
        cluster.t[index],
        cluster.ids[index],
    )


def _checked_pair(pair: tuple[int, int], known: set[int]) -> tuple[int, int]:
    """Return the pair when both ids are travelers in the current pool."""
    driver, rider = pair
    return _pool_id(driver, known), _pool_id(rider, known)


def _pool_id(value: int, known: set[int]) -> int:
    """Return ``value`` when it is a traveler id in the current pool."""
    try:
        traveler = operator.index(value)
    except TypeError as error:
        raise ValueError("assigner returned a traveler outside the pool") from error
    if traveler not in known:
        raise ValueError("assigner returned a traveler outside the pool")
    return traveler


def _reject_repeated_travelers(pairs: list[tuple[int, int]]) -> None:
    """Raise when one traveler is named in more than one seat."""
    used: set[int] = set()
    for driver, rider in pairs:
        if driver == rider or driver in used or rider in used:
            raise ValueError("assigner used a traveler more than once")
        used.add(driver)
        used.add(rider)


def _limit_drivers(cluster: TripCluster, matrix: np.ndarray, t: float, et: float) -> None:
    """Turn off drivers who depart after ``t + et``.

    The pool already holds departures in ``[t, t + horizon]``, so a row stays
    on only when that traveler departs in ``[t, t + et]``.
    """
    matrix &= cluster.t[:, None] <= t + et


def _expires_this_step(
    departures: dict[int, float],
    driver: int,
    rider: int,
    t: float,
    dt: float,
) -> bool:
    """Report whether either traveler departs in ``[t, t + dt)``."""
    limit = t + dt
    return bool(t <= departures[driver] < limit or t <= departures[rider] < limit)
