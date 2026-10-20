"""Tests for park-and-ride shareability filters."""

import numpy as np
import pytest

from carpoolsim.network_prep import RoadRouter
from carpoolsim.trip_cluster import FilterPnrAccess
from carpoolsim.trip_cluster import FilterPnrPassengerAccess
from carpoolsim.trip_cluster import FilterPnrRerouteEuclidean
from carpoolsim.trip_cluster import FilterPnrRerouteTravelTime
from carpoolsim.trip_cluster import FilterPnrSharedTravelTime
from carpoolsim.trip_cluster import FilterPnrWait
from carpoolsim.trip_cluster import PnrPlan
from carpoolsim.trip_cluster import Shareability
from carpoolsim.trip_cluster import TripCluster


def test_plan_rejects_bad_lots() -> None:
    with pytest.raises(ValueError):
        PnrPlan([0.0], [0.0], -0.1)
    with pytest.raises(ValueError):
        PnrPlan([], [], 1)
    with pytest.raises(ValueError):
        PnrPlan([0.0, 1.0], [0.0], 1)
    with pytest.raises(ValueError):
        PnrPlan(np.zeros((2, 1)), np.zeros((2, 1)), 1)


def test_access_filter_keeps_pairs_inside_r_p_including_the_exact_limit() -> None:
    # The lot is at the origin. Traveler 1 is exactly 1.5 km away, traveler 2 is just past it.
    cluster = TripCluster(
        ox=np.array([0.0, 1500.0, 1501.0]),
        oy=np.zeros(3),
        dx=np.zeros(3),
        dy=np.zeros(3),
        t=np.zeros(3),
    )
    plan = PnrPlan([0.0], [0.0], 1.5)
    share = Shareability(cluster)

    share.apply(FilterPnrAccess(plan))

    assert share.matrix.tolist() == [
        [False, True, False],
        [True, False, False],
        [False, False, False],
    ]
    assert plan.lot[0, 1] == 0
    assert plan.lot[1, 0] == 0
    assert plan.lot[0, 2] == -1


def test_access_filter_chooses_the_lot_with_the_shortest_combined_drive() -> None:
    # Lot 1 is between the origins. Lot 0 is off to the side, so the sum of drives is longer.
    cluster = TripCluster(
        ox=np.array([0.0, 4000.0]),
        oy=np.zeros(2),
        dx=np.zeros(2),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([2000.0, 2000.0], [2000.0, 0.0], 3)
    share = Shareability(cluster)

    share.apply(FilterPnrAccess(plan))

    assert share.matrix[0, 1]
    assert plan.lot[0, 1] == 1
    assert plan.lot[1, 0] == 1


def test_access_filter_breaks_lot_ties_toward_the_lower_index() -> None:
    cluster = TripCluster(
        ox=np.array([0.0, 4000.0]),
        oy=np.zeros(2),
        dx=np.zeros(2),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([1000.0, 3000.0], [0.0, 0.0], 3)
    share = Shareability(cluster)

    share.apply(FilterPnrAccess(plan))

    assert plan.lot[0, 1] == 0
    assert plan.lot[1, 0] == 0


def test_access_filter_leaves_rejected_pairs_off() -> None:
    cluster = TripCluster(
        ox=np.array([0.0, 1000.0]),
        oy=np.zeros(2),
        dx=np.zeros(2),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([0.0], [0.0], 2)
    matrix = np.array([[False, False], [True, False]])

    FilterPnrAccess(plan)(cluster, matrix)

    assert matrix.tolist() == [[False, False], [True, False]]
    assert plan.lot.tolist() == [[-1, -1], [0, -1]]


def test_reroute_filter_is_directional_and_includes_the_exact_limit() -> None:
    # Both origins are at the lot. Driver 0's direct trip is 8 km.
    # The reroute is 6 km north, then 10 km to the driver's destination: exactly twice the direct trip.
    # The reverse role is three times that driver's direct trip.
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([8000.0, 0.0]),
        dy=np.array([0.0, 6000.0]),
        t=np.zeros(2),
    )
    plan = PnrPlan([0.0], [0.0], 1)
    share = Shareability(cluster)

    share.apply(FilterPnrRerouteEuclidean(plan, 2))

    assert share.matrix.tolist() == [[False, True], [False, False]]
    assert plan.lot.tolist() == [[-1, 0], [-1, -1]]


def test_reroute_filter_picks_the_lot_that_shortens_the_driver_path() -> None:
    # Lot 0 sits on traveler 1, so access prefers it. Lot 1 sits on the way east.
    # Driver 0 is shorter through lot 1. Driver 1 is already at lot 0, so that direction keeps lot 0.
    cluster = TripCluster(
        ox=np.array([0.0, 0.0]),
        oy=np.array([0.0, 1000.0]),
        dx=np.array([10000.0, 10000.0]),
        dy=np.array([0.0, 0.0]),
        t=np.zeros(2),
    )
    plan = PnrPlan([0.0, 3000.0], [1000.0, 0.0], 4)
    share = Shareability(cluster)

    share.apply(FilterPnrAccess(plan), FilterPnrRerouteEuclidean(plan, 1.05))

    assert share.matrix.tolist() == [[False, True], [True, False]]
    assert plan.lot.tolist() == [[-1, 1], [0, -1]]


def test_reroute_filter_ignores_a_lot_beyond_r_p() -> None:
    # Lot 1 would make the driver's path exactly the direct trip, but it is 5 km from both origins.
    # The only lot inside 1 km inflates the trip by more than 1.05.
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([10000.0, 10000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([0.0, 5000.0], [1000.0, 0.0], 1)
    share = Shareability(cluster)

    share.apply(FilterPnrRerouteEuclidean(plan, 1.05))

    assert not share.matrix[0, 1]
    assert plan.lot[0, 1] == -1


def test_reroute_filter_rejects_a_driver_with_no_direct_trip() -> None:
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([0.0, 1000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([0.0], [0.0], 1)
    share = Shareability(cluster)

    share.apply(FilterPnrRerouteEuclidean(plan, 1.5))

    assert not share.matrix[0, 1]
    assert plan.lot[0, 1] == -1


def test_reroute_filter_leaves_rejected_pairs_off() -> None:
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([8000.0, 8000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([0.0], [0.0], 1)
    matrix = np.array([[False, False], [True, False]])

    FilterPnrRerouteEuclidean(plan, 1.5)(cluster, matrix)

    assert matrix.tolist() == [[False, False], [True, False]]
    assert plan.lot.tolist() == [[-1, -1], [0, -1]]


def test_negative_mu_1_is_rejected() -> None:
    with pytest.raises(ValueError):
        FilterPnrRerouteEuclidean(PnrPlan([0.0], [0.0], 1), -0.1)


def test_passenger_access_is_directional_and_includes_the_exact_limit() -> None:
    # Both solo trips are 10 km. Lot 0 is 2 km away and lot 1 is 5 km away.
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([10000.0, 10000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([2000.0, 5000.0], [0.0, 0.0], 6)
    plan.lot = np.array([[-1, 0], [1, -1]])
    share = Shareability(cluster)

    share.apply(FilterPnrPassengerAccess(plan, 0.2))

    assert share.matrix.tolist() == [[False, True], [False, False]]


def test_passenger_access_rejects_a_passenger_with_no_direct_trip() -> None:
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([10000.0, 0.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([0.0], [0.0], 1)
    plan.lot = np.array([[-1, 0], [0, -1]])
    share = Shareability(cluster)

    share.apply(FilterPnrPassengerAccess(plan, 1))

    assert not share.matrix[0, 1]


def test_passenger_access_rejects_a_pair_with_no_lot() -> None:
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([10000.0, 10000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([0.0], [0.0], 1)
    plan.lot = np.full((2, 2), -1, dtype=np.int64)
    share = Shareability(cluster)

    share.apply(FilterPnrPassengerAccess(plan, 1))

    assert not share.matrix[0, 1]


def test_passenger_access_leaves_rejected_pairs_off() -> None:
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([10000.0, 10000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([0.0], [0.0], 1)
    plan.lot = np.array([[-1, 0], [0, -1]])
    matrix = np.array([[False, False], [True, False]])

    FilterPnrPassengerAccess(plan, 1)(cluster, matrix)

    assert matrix.tolist() == [[False, False], [True, False]]


def test_filters_that_read_lots_require_a_chosen_lot(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 4.0)],
    )
    plan = PnrPlan([0.0], [0.0], 1)
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([1000.0, 1000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    matrix = np.ones((2, 2), dtype=bool)
    router = RoadRouter(nodes, edges)
    filters = [
        FilterPnrPassengerAccess(plan, 0.5),
        FilterPnrRerouteTravelTime(plan, router, 1.3),
        FilterPnrWait(plan, router, 5),
        FilterPnrSharedTravelTime(plan, router, 0.5),
    ]
    for step in filters:
        with pytest.raises(RuntimeError):
            step(cluster, matrix.copy())


def test_negative_mu_3_is_rejected() -> None:
    with pytest.raises(ValueError):
        FilterPnrPassengerAccess(PnrPlan([0.0], [0.0], 1), -0.1)


def test_reroute_time_filter_is_directional_and_includes_the_exact_limit(network_builder) -> None:
    # Driver 0's solo trip is 10 minutes. Meeting at node 2, then nodes 3 and 4, takes 13.
    # The reverse tour is 29 minutes against a 10 minute solo trip.
    nodes, edges = network_builder(
        [1, 2, 3, 4, 5],
        [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0), (3000.0, 0.0), (0.0, 1000.0)],
        [
            (1, 2, 0, 4.0),
            (2, 3, 0, 8.0),
            (3, 4, 0, 1.0),
            (1, 4, 0, 10.0),
            (5, 2, 0, 10.0),
            (2, 4, 0, 10.0),
            (4, 3, 0, 10.0),
            (5, 3, 0, 10.0),
        ],
    )
    cluster = TripCluster(
        ox=np.array([0.0, 0.0]),
        oy=np.array([0.0, 1000.0]),
        dx=np.array([3000.0, 2000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([1000.0], [0.0], 2)
    plan.lot = np.array([[-1, 0], [0, -1]])
    share = Shareability(cluster)

    share.apply(FilterPnrRerouteTravelTime(plan, RoadRouter(nodes, edges), 1.3))

    assert share.matrix.tolist() == [[False, True], [False, False]]


def test_reroute_time_filter_rejects_a_driver_with_no_solo_trip(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 4.0), (2, 1, 0, 4.0)],
    )
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([0.0, 1000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([1000.0], [0.0], 2)
    plan.lot = np.array([[-1, 0], [0, -1]])
    share = Shareability(cluster)

    share.apply(FilterPnrRerouteTravelTime(plan, RoadRouter(nodes, edges), 1.3))

    assert not share.matrix[0, 1]


def test_reroute_time_filter_rejects_a_disconnected_lot(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2, 3],
        [(0.0, 0.0), (1000.0, 0.0), (5000.0, 0.0)],
        [(1, 2, 0, 4.0), (2, 1, 0, 4.0)],
    )
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([1000.0, 1000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([5000.0], [0.0], 6)
    plan.lot = np.array([[-1, 0], [0, -1]])
    share = Shareability(cluster)

    share.apply(FilterPnrRerouteTravelTime(plan, RoadRouter(nodes, edges), 1.3))

    assert not share.matrix[0, 1]


def test_reroute_time_filter_rejects_a_pair_with_no_lot(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 4.0), (2, 1, 0, 4.0)],
    )
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([1000.0, 1000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([0.0], [0.0], 1)
    plan.lot = np.full((2, 2), -1, dtype=np.int64)
    share = Shareability(cluster)

    share.apply(FilterPnrRerouteTravelTime(plan, RoadRouter(nodes, edges), 2))

    assert not share.matrix[0, 1]


def test_reroute_time_filter_leaves_rejected_pairs_off(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 4.0), (2, 1, 0, 4.0)],
    )
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([1000.0, 1000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([0.0], [0.0], 1)
    plan.lot = np.array([[-1, 0], [0, -1]])
    matrix = np.array([[False, False], [True, False]])

    FilterPnrRerouteTravelTime(plan, RoadRouter(nodes, edges), 2)(cluster, matrix)

    assert matrix.tolist() == [[False, False], [True, False]]


def test_negative_pnr_gamma_1_is_rejected(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 4.0)],
    )
    with pytest.raises(ValueError):
        FilterPnrRerouteTravelTime(PnrPlan([0.0], [0.0], 1), RoadRouter(nodes, edges), -0.1)


def test_wait_filter_includes_the_exact_limit(network_builder) -> None:
    # The driver reaches the lot in 4 minutes and the passenger in 10. The difference is 6.
    nodes, edges = network_builder(
        [1, 2, 5],
        [(0.0, 0.0), (1000.0, 0.0), (0.0, 1000.0)],
        [(1, 2, 0, 4.0), (5, 2, 0, 10.0)],
    )
    cluster = TripCluster(
        ox=np.array([0.0, 0.0]),
        oy=np.array([0.0, 1000.0]),
        dx=np.zeros(2),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([1000.0], [0.0], 2)
    plan.lot = np.array([[-1, 0], [0, -1]])
    share = Shareability(cluster)

    share.apply(FilterPnrWait(plan, RoadRouter(nodes, edges), 6))

    assert share.matrix[0, 1]
    assert share.matrix[1, 0]


def test_wait_filter_is_directional_when_each_role_uses_its_own_lot(network_builder) -> None:
    # Lot 0 takes 4 and 10 minutes, a wait of 6. Lot 1 takes 1 and 6 minutes, a wait of 5.
    nodes, edges = network_builder(
        [1, 2, 5, 6],
        [(0.0, 0.0), (1000.0, 0.0), (0.0, 1000.0), (5000.0, 0.0)],
        [
            (1, 2, 0, 4.0),
            (5, 2, 0, 10.0),
            (1, 6, 0, 1.0),
            (5, 6, 0, 6.0),
        ],
    )
    cluster = TripCluster(
        ox=np.array([0.0, 0.0]),
        oy=np.array([0.0, 1000.0]),
        dx=np.zeros(2),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([1000.0, 5000.0], [0.0, 0.0], 6)
    plan.lot = np.array([[-1, 0], [1, -1]])
    share = Shareability(cluster)

    share.apply(FilterPnrWait(plan, RoadRouter(nodes, edges), 5))

    assert share.matrix.tolist() == [[False, False], [True, False]]


def test_wait_filter_rejects_an_unreachable_lot(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2, 3],
        [(0.0, 0.0), (1000.0, 0.0), (5000.0, 0.0)],
        [(1, 2, 0, 4.0)],
    )
    cluster = TripCluster(
        ox=np.array([0.0, 1000.0]),
        oy=np.zeros(2),
        dx=np.zeros(2),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([5000.0], [0.0], 6)
    plan.lot = np.array([[-1, 0], [0, -1]])
    share = Shareability(cluster)

    share.apply(FilterPnrWait(plan, RoadRouter(nodes, edges), 30))

    assert not share.matrix[0, 1]


def test_wait_filter_leaves_rejected_pairs_off(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 4.0), (2, 1, 0, 4.0)],
    )
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.zeros(2),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([0.0], [0.0], 1)
    plan.lot = np.array([[-1, 0], [0, -1]])
    matrix = np.array([[False, False], [True, False]])

    FilterPnrWait(plan, RoadRouter(nodes, edges), 0)(cluster, matrix)

    assert matrix.tolist() == [[False, False], [True, False]]


def test_negative_pnr_delta2_is_rejected(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 4.0)],
    )
    with pytest.raises(ValueError):
        FilterPnrWait(PnrPlan([0.0], [0.0], 1), RoadRouter(nodes, edges), -0.1)


def test_shared_time_filter_is_directional_and_includes_the_exact_limit(network_builder) -> None:
    # Driver 0's tour is 1 + 8 + 1 minutes, so the shared 8 minutes are exactly 0.8.
    # The reverse shared leg is 9 of the 29 minutes on that tour.
    nodes, edges = network_builder(
        [1, 2, 3, 4, 5],
        [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0), (3000.0, 0.0), (0.0, 1000.0)],
        [
            (1, 2, 0, 1.0),
            (2, 3, 0, 8.0),
            (3, 4, 0, 1.0),
            (5, 2, 0, 10.0),
            (2, 4, 0, 10.0),
            (4, 3, 0, 10.0),
        ],
    )
    cluster = TripCluster(
        ox=np.array([0.0, 0.0]),
        oy=np.array([0.0, 1000.0]),
        dx=np.array([3000.0, 2000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([1000.0], [0.0], 2)
    plan.lot = np.array([[-1, 0], [0, -1]])
    share = Shareability(cluster)

    share.apply(FilterPnrSharedTravelTime(plan, RoadRouter(nodes, edges), 0.8))

    assert share.matrix.tolist() == [[False, True], [False, False]]


def test_shared_time_filter_rejects_a_tour_with_no_travel(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 4.0), (2, 1, 0, 4.0)],
    )
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.zeros(2),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([0.0], [0.0], 1)
    plan.lot = np.array([[-1, 0], [0, -1]])
    share = Shareability(cluster)

    share.apply(FilterPnrSharedTravelTime(plan, RoadRouter(nodes, edges), 0.8))

    assert not share.matrix[0, 1]


def test_shared_time_filter_rejects_a_disconnected_pair(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2, 3],
        [(0.0, 0.0), (1000.0, 0.0), (5000.0, 0.0)],
        [(1, 2, 0, 4.0), (2, 1, 0, 4.0)],
    )
    cluster = TripCluster(
        ox=np.array([0.0, 5000.0]),
        oy=np.zeros(2),
        dx=np.array([1000.0, 5000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([0.0], [0.0], 6)
    plan.lot = np.array([[-1, 0], [0, -1]])
    share = Shareability(cluster)

    share.apply(FilterPnrSharedTravelTime(plan, RoadRouter(nodes, edges), 0.8))

    assert not share.matrix[0, 1]


def test_shared_time_filter_leaves_rejected_pairs_off(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 4.0), (2, 1, 0, 4.0)],
    )
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([1000.0, 1000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    plan = PnrPlan([0.0], [0.0], 1)
    plan.lot = np.array([[-1, 0], [0, -1]])
    matrix = np.array([[False, False], [True, False]])

    FilterPnrSharedTravelTime(plan, RoadRouter(nodes, edges), 0.5)(cluster, matrix)

    assert matrix.tolist() == [[False, False], [True, False]]


def test_pnr_gamma_2_outside_zero_to_one_is_rejected(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 4.0)],
    )
    plan = PnrPlan([0.0], [0.0], 1)
    router = RoadRouter(nodes, edges)
    with pytest.raises(ValueError):
        FilterPnrSharedTravelTime(plan, router, -0.1)
    with pytest.raises(ValueError):
        FilterPnrSharedTravelTime(plan, router, 1.1)
