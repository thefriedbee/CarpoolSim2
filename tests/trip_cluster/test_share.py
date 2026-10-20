"""Tests for the shareability matrix and the departure-time filter."""

import numpy as np
import pytest

from carpoolsim.trip_cluster import FilterDepartureTime
from carpoolsim.trip_cluster import FilterDestinationEuclideanDistance
from carpoolsim.trip_cluster import FilterDropoffEuclideanDistance
from carpoolsim.trip_cluster import FilterOriginEuclideanDistance
from carpoolsim.trip_cluster import FilterPassengerWaitTime
from carpoolsim.network_prep import RoadRouter
from carpoolsim.trip_cluster import FilterRerouteEuclideanDistance
from carpoolsim.trip_cluster import FilterRerouteTravelTime
from carpoolsim.trip_cluster import FilterSharedTravelTime
from carpoolsim.trip_cluster import Shareability
from carpoolsim.trip_cluster import TripCluster


def cluster_at(*minutes: float) -> TripCluster:
    n = len(minutes)
    zeros = np.zeros(n)
    return TripCluster(ox=zeros, oy=zeros, dx=zeros.copy(), dy=zeros.copy(), t=np.array(minutes))


def test_new_matrix_rejects_the_diagonal_only() -> None:
    share = Shareability(cluster_at(0, 10, 40))

    assert share.matrix.tolist() == [
        [False, True, True],
        [True, False, True],
        [True, True, False],
    ]


def test_departure_filter_keeps_pairs_inside_delta1() -> None:
    share = Shareability(cluster_at(0, 10, 25))

    share.apply(FilterDepartureTime(15))

    assert share.matrix.tolist() == [
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ]


def test_departure_filter_includes_the_exact_limit() -> None:
    share = Shareability(cluster_at(0, 15))

    share.apply(FilterDepartureTime(15))

    assert share.matrix[0, 1]
    assert share.matrix[1, 0]


def test_departure_filter_leaves_rejected_pairs_off() -> None:
    cluster = cluster_at(0, 5)
    matrix = np.array([[False, False], [True, False]])

    FilterDepartureTime(10)(cluster, matrix)

    assert matrix.tolist() == [[False, False], [True, False]]


def test_negative_delta1_is_rejected() -> None:
    with pytest.raises(ValueError):
        FilterDepartureTime(-1)


def test_origin_filter_keeps_pairs_inside_r1() -> None:
    cluster = TripCluster(
        ox=np.array([0.0, 1000.0, 3000.0]),
        oy=np.zeros(3),
        dx=np.zeros(3),
        dy=np.zeros(3),
        t=np.zeros(3),
    )
    share = Shareability(cluster)

    share.apply(FilterOriginEuclideanDistance(2))

    assert share.matrix.tolist() == [
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ]


def test_destination_filter_keeps_pairs_inside_r2() -> None:
    cluster = TripCluster(
        ox=np.zeros(3),
        oy=np.zeros(3),
        dx=np.array([0.0, 1000.0, 3000.0]),
        dy=np.zeros(3),
        t=np.zeros(3),
    )
    share = Shareability(cluster)

    share.apply(FilterDestinationEuclideanDistance(2))

    assert share.matrix.tolist() == [
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ]


def test_distance_filters_include_the_exact_limit() -> None:
    cluster = TripCluster(
        ox=np.array([0.0, 1500.0]),
        oy=np.zeros(2),
        dx=np.array([0.0, 2500.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    share = Shareability(cluster)

    share.apply(FilterOriginEuclideanDistance(1.5), FilterDestinationEuclideanDistance(2.5))

    assert share.matrix[0, 1]
    assert share.matrix[1, 0]


def test_negative_distance_limits_are_rejected() -> None:
    with pytest.raises(ValueError):
        FilterOriginEuclideanDistance(-0.1)
    with pytest.raises(ValueError):
        FilterDestinationEuclideanDistance(-0.1)


def test_reroute_filter_is_directional_and_includes_the_exact_limit() -> None:
    # Driver 0 travels 8 m east. Picking up 1 inflates that trip by exactly 1.5.
    # The reverse role inflates traveler 1's trip by 5.
    cluster = TripCluster(
        ox=np.array([0.0, 0.0]),
        oy=np.array([0.0, 0.0]),
        dx=np.array([8.0, 0.0]),
        dy=np.array([0.0, 10.0 / 3.0]),
        t=np.zeros(2),
    )
    share = Shareability(cluster)

    share.apply(FilterRerouteEuclideanDistance(1.5))

    assert share.matrix.tolist() == [[False, True], [False, False]]


def test_reroute_filter_rejects_a_driver_with_no_direct_trip() -> None:
    cluster = TripCluster(
        ox=np.array([0.0, 0.0]),
        oy=np.zeros(2),
        dx=np.array([0.0, 1000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    share = Shareability(cluster)

    share.apply(FilterRerouteEuclideanDistance(1.5))

    assert not share.matrix[0, 1]


def test_reroute_filter_leaves_rejected_pairs_off() -> None:
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([1000.0, 1000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    matrix = np.array([[False, False], [True, False]])

    FilterRerouteEuclideanDistance(1.5)(cluster, matrix)

    assert matrix.tolist() == [[False, False], [True, False]]


def test_negative_mu_1_is_rejected() -> None:
    with pytest.raises(ValueError):
        FilterRerouteEuclideanDistance(-0.1)


def test_dropoff_filter_is_directional_and_includes_the_exact_limit() -> None:
    # The last leg is 2 m. That is 0.2 of driver 0's 10 m trip and 0.25 of driver 1's 8 m trip.
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([10.0, 8.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    share = Shareability(cluster)

    share.apply(FilterDropoffEuclideanDistance(0.2))

    assert share.matrix.tolist() == [[False, True], [False, False]]


def test_dropoff_filter_rejects_a_driver_with_no_direct_trip() -> None:
    cluster = TripCluster(
        ox=np.array([0.0, 0.0]),
        oy=np.zeros(2),
        dx=np.array([0.0, 1000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    share = Shareability(cluster)

    share.apply(FilterDropoffEuclideanDistance(0.2))

    assert not share.matrix[0, 1]


def test_dropoff_filter_leaves_rejected_pairs_off() -> None:
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.array([1000.0, 1000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    matrix = np.array([[False, False], [True, False]])

    FilterDropoffEuclideanDistance(0.2)(cluster, matrix)

    assert matrix.tolist() == [[False, False], [True, False]]


def test_negative_mu_2_is_rejected() -> None:
    with pytest.raises(ValueError):
        FilterDropoffEuclideanDistance(-0.1)


def test_reroute_time_filter_is_directional_and_includes_the_exact_limit(network_builder) -> None:
    # Node rows 0, 1, 2. Driver 0's solo trip is 10 minutes.
    # Picking up 1 takes 13 minutes, exactly 1.3 times that solo trip.
    nodes, edges = network_builder(
        [1, 2, 3],
        [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)],
        [
            (1, 2, 0, 3.0),
            (2, 1, 0, 3.0),
            (2, 3, 0, 10.0),
            (3, 2, 0, 10.0),
            (1, 3, 0, 10.0),
            (3, 1, 0, 10.0),
        ],
    )
    cluster = TripCluster(
        ox=np.array([0.0, 0.0]),
        oy=np.zeros(2),
        dx=np.array([2000.0, 1000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    share = Shareability(cluster)

    share.apply(FilterRerouteTravelTime(RoadRouter(nodes, edges), 1.3))

    assert share.matrix.tolist() == [[False, True], [False, False]]


def test_reroute_time_filter_rejects_a_driver_with_no_solo_trip(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 4.0), (2, 1, 0, 4.0)],
    )
    cluster = TripCluster(
        ox=np.array([0.0, 0.0]),
        oy=np.zeros(2),
        dx=np.array([0.0, 1000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    share = Shareability(cluster)

    share.apply(FilterRerouteTravelTime(RoadRouter(nodes, edges), 1.3))

    assert not share.matrix[0, 1]


def test_reroute_time_filter_rejects_a_disconnected_passenger(network_builder) -> None:
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
    share = Shareability(cluster)

    share.apply(FilterRerouteTravelTime(RoadRouter(nodes, edges), 1.3))

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
    matrix = np.array([[False, False], [True, False]])

    FilterRerouteTravelTime(RoadRouter(nodes, edges), 1.3)(cluster, matrix)

    assert matrix.tolist() == [[False, False], [True, False]]


def test_passenger_wait_is_directional_and_includes_the_exact_limit(network_builder) -> None:
    # The drive from 0 to 1 takes 10 minutes, and the return takes 30.
    # Traveler 1 waits exactly 10 minutes for traveler 0. The reverse wait is 30.
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 10.0), (2, 1, 0, 30.0)],
    )
    cluster = TripCluster(
        ox=np.array([0.0, 1000.0]),
        oy=np.zeros(2),
        dx=np.array([0.0, 1000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    share = Shareability(cluster)

    share.apply(FilterPassengerWaitTime(RoadRouter(nodes, edges), 10))

    assert share.matrix.tolist() == [[False, True], [False, False]]


def test_passenger_leaves_early_instead_of_making_the_driver_wait(network_builder) -> None:
    # Traveler 0 reaches traveler 1 ten minutes before traveler 1 planned to leave.
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 10.0), (2, 1, 0, 10.0)],
    )
    cluster = TripCluster(
        ox=np.array([0.0, 1000.0]),
        oy=np.zeros(2),
        dx=np.zeros(2),
        dy=np.zeros(2),
        t=np.array([0.0, 20.0]),
    )
    share = Shareability(cluster)

    share.apply(FilterPassengerWaitTime(RoadRouter(nodes, edges), 0))

    assert share.matrix[0, 1]
    assert not share.matrix[1, 0]


def test_passenger_wait_rejects_an_unreachable_origin(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2, 3],
        [(0.0, 0.0), (1000.0, 0.0), (5000.0, 0.0)],
        [(1, 2, 0, 4.0), (2, 1, 0, 4.0)],
    )
    cluster = TripCluster(
        ox=np.array([0.0, 5000.0]),
        oy=np.zeros(2),
        dx=np.zeros(2),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    share = Shareability(cluster)

    share.apply(FilterPassengerWaitTime(RoadRouter(nodes, edges), 30))

    assert not share.matrix[0, 1]


def test_passenger_wait_leaves_rejected_pairs_off(network_builder) -> None:
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
    matrix = np.array([[False, False], [True, False]])

    FilterPassengerWaitTime(RoadRouter(nodes, edges), 0)(cluster, matrix)

    assert matrix.tolist() == [[False, False], [True, False]]


def test_negative_delta2_is_rejected(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 4.0)],
    )
    with pytest.raises(ValueError):
        FilterPassengerWaitTime(RoadRouter(nodes, edges), -0.1)


def test_shared_time_filter_is_directional_and_includes_the_exact_limit(network_builder) -> None:
    # Driver 0's tour is 1 + 8 + 1 minutes, so the shared 8 minutes are exactly 0.8.
    # The reverse tour is mostly the drive to the pickup and after the drop-off.
    nodes, edges = network_builder(
        [1, 2, 3, 4],
        [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0), (3000.0, 0.0)],
        [
            (1, 2, 0, 1.0),
            (2, 1, 0, 10.0),
            (2, 3, 0, 8.0),
            (3, 2, 0, 8.0),
            (3, 4, 0, 1.0),
            (4, 3, 0, 10.0),
            (1, 4, 0, 1.0),
            (4, 1, 0, 10.0),
        ],
    )
    cluster = TripCluster(
        ox=np.array([0.0, 1000.0]),
        oy=np.zeros(2),
        dx=np.array([3000.0, 2000.0]),
        dy=np.zeros(2),
        t=np.zeros(2),
    )
    share = Shareability(cluster)

    share.apply(FilterSharedTravelTime(RoadRouter(nodes, edges), 0.8))

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
    share = Shareability(cluster)

    share.apply(FilterSharedTravelTime(RoadRouter(nodes, edges), 0.8))

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
    share = Shareability(cluster)

    share.apply(FilterSharedTravelTime(RoadRouter(nodes, edges), 0.8))

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
    matrix = np.array([[False, False], [True, False]])

    FilterSharedTravelTime(RoadRouter(nodes, edges), 0.8)(cluster, matrix)

    assert matrix.tolist() == [[False, False], [True, False]]


def test_gamma_2_outside_zero_to_one_is_rejected(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 4.0)],
    )
    router = RoadRouter(nodes, edges)
    with pytest.raises(ValueError):
        FilterSharedTravelTime(router, -0.1)
    with pytest.raises(ValueError):
        FilterSharedTravelTime(router, 1.1)


def test_negative_gamma_1_is_rejected(network_builder) -> None:
    nodes, edges = network_builder(
        [1, 2],
        [(0.0, 0.0), (1000.0, 0.0)],
        [(1, 2, 0, 4.0)],
    )
    with pytest.raises(ValueError):
        FilterRerouteTravelTime(RoadRouter(nodes, edges), -0.1)
