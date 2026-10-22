"""Tests for the rolling departure window."""

import numpy as np
import pytest

from carpoolsim.optimizer import GreedyAssigner
from carpoolsim.simulator import Simulator
from carpoolsim.trip_cluster import FilterDepartureTime
from carpoolsim.trip_cluster import TripCluster


def cluster_at(*minutes: float) -> TripCluster:
    n = len(minutes)
    zeros = np.zeros(n)
    return TripCluster(
        ox=zeros,
        oy=zeros,
        dx=zeros.copy(),
        dy=zeros.copy(),
        t=np.array(minutes, dtype=float),
    )


class RecordPools:
    """Remember the departure times passed to each filter call."""

    def __init__(self) -> None:
        self.pools: list[list[float]] = []

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        self.pools.append(cluster.t.tolist())


def no_pairs(matrix: np.ndarray, ids: np.ndarray) -> list[tuple[int, int]]:
    """Leave every traveler unmatched."""
    return []


def test_step_must_be_positive_and_fit_inside_the_horizon() -> None:
    cluster = cluster_at(0)

    with pytest.raises(ValueError, match="dt must be positive"):
        Simulator(cluster, GreedyAssigner(), dt=0)
    with pytest.raises(ValueError, match="horizon must be at least dt"):
        Simulator(cluster, GreedyAssigner(), horizon=5, dt=10)
    with pytest.raises(ValueError, match="commit must be 'window' or 'step'"):
        Simulator(cluster, GreedyAssigner(), commit="later")

    Simulator(cluster, GreedyAssigner(), horizon=5, dt=5)


def test_a_trip_enters_at_the_horizon_and_leaves_after_its_departure() -> None:
    record = RecordPools()
    simulator = Simulator(cluster_at(0, 30), no_pairs, filters=(record,), horizon=15, dt=5)

    simulator.run()

    assert record.pools == [[0.0]] * 4 + [[30.0]] * 4
    assert [(solo.traveler, solo.t) for solo in simulator.solos] == [(0, 5.0), (1, 35.0)]
    assert simulator.matches == []


def test_window_commit_removes_both_members_on_the_first_shared_tick() -> None:
    simulator = Simulator(cluster_at(0, 10), GreedyAssigner(), horizon=15, dt=5)

    simulator.run()

    assert [(match.driver, match.rider, match.t) for match in simulator.matches] == [(0, 1, -5.0)]
    assert simulator.solos == []


def test_step_commit_waits_until_one_traveler_is_about_to_leave() -> None:
    simulator = Simulator(
        cluster_at(0, 10),
        GreedyAssigner(),
        horizon=15,
        dt=5,
        commit="step",
    )

    simulator.run()

    assert [(match.driver, match.rider, match.t) for match in simulator.matches] == [(0, 1, 0.0)]
    assert simulator.solos == []


def test_step_commit_keeps_a_later_partner() -> None:
    simulator = Simulator(
        cluster_at(0, 12),
        GreedyAssigner(),
        horizon=15,
        dt=5,
        commit="step",
    )

    simulator.run()

    assert [(match.driver, match.rider, match.t) for match in simulator.matches] == [(0, 1, 0.0)]
    assert simulator.solos == []


def test_local_order_follows_departure_then_maps_back_to_row_ids() -> None:
    def earliest_drives_the_next(matrix: np.ndarray, ids: np.ndarray) -> list[tuple[int, int]]:
        if matrix.shape[0] < 2:
            return []
        return [(int(ids[0]), int(ids[1]))]

    simulator = Simulator(cluster_at(10, 0), earliest_drives_the_next, horizon=15, dt=5)

    simulator.run()

    assert [(match.driver, match.rider, match.t) for match in simulator.matches] == [(1, 0, -5.0)]
    assert simulator.solos == []


def test_an_unmatched_trip_remains_until_the_clock_passes_its_departure() -> None:
    record = RecordPools()
    simulator = Simulator(cluster_at(0, 10, 40), no_pairs, filters=(record,), horizon=15, dt=5)

    simulator.run()

    seen = [pool for pool in record.pools if 0.0 in pool]
    assert len(seen) == 4
    assert 10.0 in seen[-1]
    assert [(solo.traveler, solo.t) for solo in simulator.solos] == [(0, 5.0), (1, 15.0), (2, 45.0)]


def test_every_traveler_is_matched_or_solo_exactly_once() -> None:
    simulator = Simulator(cluster_at(0, 5, 50), GreedyAssigner(), horizon=15, dt=5)

    simulator.run()

    matched = [match.driver for match in simulator.matches] + [match.rider for match in simulator.matches]
    solo = [solo.traveler for solo in simulator.solos]
    assert sorted(matched + solo) == [0, 1, 2]
    assert set(matched).isdisjoint(solo)


def test_departure_filter_drops_pairs_inside_the_window() -> None:
    simulator = Simulator(
        cluster_at(0, 8, 10),
        GreedyAssigner(),
        filters=(FilterDepartureTime(5),),
        horizon=15,
        dt=5,
    )

    simulator.run()

    assert [(match.driver, match.rider, match.t) for match in simulator.matches] == [(1, 2, -5.0)]
    assert [(solo.traveler, solo.t) for solo in simulator.solos] == [(0, 5.0)]


def test_a_repeated_traveler_is_rejected() -> None:
    def repeat(matrix: np.ndarray, ids: np.ndarray) -> list[tuple[int, int]]:
        if matrix.shape[0] < 3:
            return []
        return [(int(ids[0]), int(ids[1])), (int(ids[1]), int(ids[2]))]

    simulator = Simulator(cluster_at(0, 0, 0), repeat, horizon=15, dt=5)

    with pytest.raises(ValueError, match="assigner used a traveler more than once"):
        simulator.run()
    assert simulator.matches == []


def test_an_id_outside_the_pool_is_rejected() -> None:
    def outside(matrix: np.ndarray, ids: np.ndarray) -> list[tuple[int, int]]:
        if matrix.shape[0] < 2:
            return []
        return [(int(ids[0]), 5)]

    simulator = Simulator(cluster_at(0, 1), outside, horizon=15, dt=5)

    with pytest.raises(ValueError, match="assigner returned a traveler outside the pool"):
        simulator.run()


def test_an_empty_day_records_nothing() -> None:
    simulator = Simulator(cluster_at(), GreedyAssigner())

    simulator.run()

    assert simulator.matches == []
    assert simulator.solos == []


def marked(*xs: float) -> TripCluster:
    n = len(xs)
    return TripCluster(
        ox=np.array(xs, dtype=float),
        oy=np.zeros(n),
        dx=np.zeros(n),
        dy=np.zeros(n),
        t=np.zeros(n),
    )


def allow(driver_x: float, rider_x: float):
    """Keep only the ordered pair whose origins carry these marker values."""

    def step(cluster: TripCluster, matrix: np.ndarray) -> None:
        keep = (cluster.ox[:, None] == driver_x) & (cluster.ox[None, :] == rider_x)
        matrix &= keep

    return step


class RecordOx:
    """Remember the origin markers passed to a mode."""

    def __init__(self) -> None:
        self.seen: list[list[float]] = []

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        self.seen.append(cluster.ox.tolist())


def test_the_first_mode_keeps_travelers_from_the_later_mode() -> None:
    direct_first = Simulator(
        marked(0, 1, 2),
        GreedyAssigner(),
        modes=(("direct", (allow(0, 1),)), ("pnr", (allow(1, 2),))),
        horizon=15,
        dt=5,
    )
    pnr_first = Simulator(
        marked(0, 1, 2),
        GreedyAssigner(),
        modes=(("pnr", (allow(1, 2),)), ("direct", (allow(0, 1),))),
        horizon=15,
        dt=5,
    )

    direct_first.run()
    pnr_first.run()

    assert [(match.driver, match.rider, match.mode) for match in direct_first.matches] == [
        (0, 1, "direct")
    ]
    assert [(solo.traveler, solo.t) for solo in direct_first.solos] == [(2, 5.0)]
    assert [(match.driver, match.rider, match.mode) for match in pnr_first.matches] == [
        (1, 2, "pnr")
    ]
    assert [(solo.traveler, solo.t) for solo in pnr_first.solos] == [(0, 5.0)]


def test_a_later_mode_matches_only_the_travelers_left_behind() -> None:
    seen = RecordOx()
    simulator = Simulator(
        marked(0, 1, 2, 3),
        GreedyAssigner(),
        modes=(
            ("direct", (allow(0, 1),)),
            ("pnr", (seen, allow(2, 3))),
        ),
        horizon=15,
        dt=5,
    )

    simulator.run()

    assert [(match.driver, match.rider, match.mode) for match in simulator.matches] == [
        (0, 1, "direct"),
        (2, 3, "pnr"),
    ]
    assert simulator.solos == []
    assert seen.seen == [[2.0, 3.0]]


def test_a_later_mode_can_match_a_pair_the_earlier_mode_rejected() -> None:
    def reject(cluster: TripCluster, matrix: np.ndarray) -> None:
        matrix[:] = False

    simulator = Simulator(
        cluster_at(0, 10),
        GreedyAssigner(),
        modes=(("direct", (reject,)), ("pnr", ())),
        horizon=15,
        dt=5,
        commit="step",
    )

    simulator.run()

    assert [(match.driver, match.rider, match.t, match.mode) for match in simulator.matches] == [
        (0, 1, 0.0, "pnr")
    ]
    assert simulator.solos == []


def test_mode_names_are_required_once_and_not_mixed_with_filters() -> None:
    with pytest.raises(ValueError, match="at least one mode is required"):
        Simulator(cluster_at(0), GreedyAssigner(), modes=())
    with pytest.raises(ValueError, match="mode name must be a non-empty string"):
        Simulator(cluster_at(0), GreedyAssigner(), modes=(("", ()),))
    with pytest.raises(ValueError, match="mode names must be unique"):
        Simulator(cluster_at(0), GreedyAssigner(), modes=(("direct", ()), ("direct", ())))
    with pytest.raises(ValueError, match="pass filters or modes, not both"):
        Simulator(
            cluster_at(0),
            GreedyAssigner(),
            filters=(RecordPools(),),
            modes=(("direct", ()),),
        )


def test_matches_and_solos_record_traveler_ids() -> None:
    paired = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.zeros(2),
        dy=np.zeros(2),
        t=np.array([0.0, 10.0]),
        ids=np.array([10, 20]),
    )
    alone = TripCluster(
        ox=np.zeros(1),
        oy=np.zeros(1),
        dx=np.zeros(1),
        dy=np.zeros(1),
        t=np.array([0.0]),
        ids=np.array([8]),
    )
    paired_run = Simulator(paired, GreedyAssigner(), horizon=15, dt=5, commit="step")
    alone_run = Simulator(alone, GreedyAssigner(), horizon=15, dt=5)

    paired_run.run()
    alone_run.run()

    assert [(match.driver, match.rider, match.t) for match in paired_run.matches] == [
        (10, 20, 0.0)
    ]
    assert paired_run.solos == []
    assert [(solo.traveler, solo.t) for solo in alone_run.solos] == [(8, 5.0)]


def test_a_second_run_replaces_the_previous_result() -> None:
    simulator = Simulator(cluster_at(0, 10), GreedyAssigner(), horizon=15, dt=5)

    simulator.run()
    simulator.run()

    assert [(match.driver, match.rider) for match in simulator.matches] == [(0, 1)]
    assert simulator.solos == []


def test_et_must_sit_between_the_step_and_the_horizon() -> None:
    cluster = cluster_at(0)

    with pytest.raises(ValueError, match="et must be at least dt"):
        Simulator(cluster, GreedyAssigner(), horizon=15, dt=5, et=4)
    with pytest.raises(ValueError, match="et must be at most horizon"):
        Simulator(cluster, GreedyAssigner(), horizon=15, dt=5, et=16)

    Simulator(cluster, GreedyAssigner(), horizon=15, dt=5, et=5)
    Simulator(cluster, GreedyAssigner(), horizon=15, dt=5, et=15)


def later_departure_drives(cluster: TripCluster, matrix: np.ndarray) -> None:
    """Keep a pair only when the driver departs after the rider."""
    matrix &= cluster.t[:, None] > cluster.t[None, :]


def test_et_holds_a_later_driver_until_that_departure_enters_the_slice() -> None:
    held = Simulator(
        cluster_at(0, 4),
        GreedyAssigner(),
        filters=(later_departure_drives,),
        horizon=15,
        dt=5,
        et=5,
    )
    open_drivers = Simulator(
        cluster_at(0, 4),
        GreedyAssigner(),
        filters=(later_departure_drives,),
        horizon=15,
        dt=5,
    )

    held.run()
    open_drivers.run()

    assert [(match.driver, match.rider, match.t) for match in held.matches] == [(1, 0, 0.0)]
    assert held.solos == []
    assert [(match.driver, match.rider, match.t) for match in open_drivers.matches] == [
        (1, 0, -10.0)
    ]
    assert open_drivers.solos == []


class RecordMatrix:
    """Remember the departure times and matrix passed to a filter."""

    def __init__(self) -> None:
        self.seen: list[tuple[list[float], list[list[bool]]]] = []

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        self.seen.append((cluster.t.tolist(), matrix.tolist()))


def test_et_lets_an_early_driver_pick_up_a_passenger_outside_the_slice() -> None:
    record = RecordMatrix()
    simulator = Simulator(
        cluster_at(0, 12),
        GreedyAssigner(),
        filters=(record,),
        horizon=15,
        dt=5,
        et=5,
    )

    simulator.run()

    assert [(match.driver, match.rider, match.t) for match in simulator.matches] == [(0, 1, 0.0)]
    assert simulator.solos == []
    shared = [matrix for departures, matrix in record.seen if departures == [0.0, 12.0]]
    assert shared == [[[False, True], [False, False]]]


def test_step_commit_defers_a_driver_who_leaves_after_this_step() -> None:
    window = Simulator(
        cluster_at(10, 14),
        GreedyAssigner(),
        horizon=15,
        dt=5,
        et=10,
    )
    step = Simulator(
        cluster_at(10, 14),
        GreedyAssigner(),
        horizon=15,
        dt=5,
        et=10,
        commit="step",
    )

    window.run()
    step.run()

    assert [(match.driver, match.rider, match.t) for match in window.matches] == [(0, 1, 0.0)]
    assert window.solos == []
    assert [(match.driver, match.rider, match.t) for match in step.matches] == [(0, 1, 10.0)]
    assert step.solos == []


class RecordIds:
    """Remember the traveler ids passed to each filter call."""

    def __init__(self) -> None:
        self.seen: list[list[int]] = []

    def __call__(self, cluster: TripCluster, matrix: np.ndarray) -> None:
        self.seen.append([int(trip_id) for trip_id in cluster.ids])


def pair_ends(matrix: np.ndarray, ids: np.ndarray) -> list[tuple[int, int]]:
    """Pair the first traveler in this matrix with the last."""
    if ids.shape[0] < 2:
        return []
    return [(int(ids[0]), int(ids[-1]))]


def test_max_size_must_be_an_integer_of_at_least_two() -> None:
    cluster = cluster_at(0)

    with pytest.raises(ValueError, match="max_size must be at least 2"):
        Simulator(cluster, GreedyAssigner(), max_size=1)
    with pytest.raises(ValueError, match="max_size must be an integer"):
        Simulator(cluster, GreedyAssigner(), max_size=True)


def test_max_size_filters_each_spatial_group_on_its_own() -> None:
    record = RecordIds()
    simulator = Simulator(
        cluster_at(0, 0, 0, 0),
        GreedyAssigner(),
        filters=(record,),
        horizon=15,
        dt=5,
        max_size=2,
    )

    simulator.run()

    assert record.seen == [[0, 1], [2, 3]]
    assert [(match.driver, match.rider) for match in simulator.matches] == [(0, 1), (2, 3)]
    assert simulator.solos == []


def test_max_size_keeps_a_pair_inside_its_own_group() -> None:
    origins = np.array([0.0, 0.0, 50_000.0, 50_000.0])
    cluster = TripCluster(origins, np.zeros(4), origins.copy(), np.zeros(4), np.zeros(4))
    open_pool = Simulator(cluster, pair_ends, horizon=15, dt=5)
    grouped = Simulator(cluster, pair_ends, horizon=15, dt=5, max_size=2)

    open_pool.run()
    grouped.run()

    assert [(match.driver, match.rider) for match in open_pool.matches] == [(0, 3), (1, 2)]
    assert [(match.driver, match.rider) for match in grouped.matches] == [(0, 1), (2, 3)]
    assert grouped.solos == []
