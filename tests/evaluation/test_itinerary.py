"""Tests for direct and park-and-ride itineraries."""

import numpy as np
import pytest

from carpoolsim.network_prep import RoadRouter
from carpoolsim.trip_cluster import TripCluster
from evaluation.itinerary import direct_itinerary
from evaluation.itinerary import pnr_itinerary
from evaluation.itinerary import solo_leg
from evaluation.itinerary import solo_minutes
from tests.conftest import Network


def line_cluster(*trips: tuple[int, float, float, float]) -> TripCluster:
    """Trips on the six-node line. Each trip is id, origin x, destination x, departure."""
    ids = [trip[0] for trip in trips]
    return TripCluster(
        ox=np.array([trip[1] for trip in trips]),
        oy=np.zeros(len(trips)),
        dx=np.array([trip[2] for trip in trips]),
        dy=np.zeros(len(trips)),
        t=np.array([trip[3] for trip in trips]),
        ids=np.array(ids),
    )


def test_solo_leg_keeps_the_routed_path(shortcut_network: Network) -> None:
    nodes, edges = shortcut_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster((42, 0.0, 2000.0, 0.0))

    leg = solo_leg(router, cluster, 42)

    assert (leg.who, leg.name) == ("solo", "solo")
    assert leg.edge_rows == [1, 2]
    assert leg.start == (0.0, 0.0)
    assert leg.end == (2000.0, 0.0)
    assert leg.travel_min == pytest.approx(2.0)
    assert solo_minutes(router, cluster, 42) == pytest.approx(leg.travel_min)


def test_direct_itinerary_counts_the_detour_and_the_wait(line_network: Network) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster((10, 0.0, 4000.0, 0.0), (11, 1000.0, 3000.0, 0.0))

    trip = direct_itinerary(router, cluster, 10, 11)

    assert trip.lot_id is None
    assert trip.driver_solo_min == pytest.approx(4.0)
    assert trip.rider_solo_min == pytest.approx(2.0)
    assert trip.minutes("driver", "pickup") == pytest.approx(1.0)
    assert trip.shared_min == pytest.approx(2.0)
    assert trip.minutes("driver", "dropoff") == pytest.approx(1.0)
    assert trip.driver_tour_min == pytest.approx(4.0)
    assert trip.driver_ratio == pytest.approx(1.0)
    assert trip.shared_share == pytest.approx(0.5)
    assert trip.rider_wait_min == pytest.approx(1.0)
    assert trip.driver_wait_min == pytest.approx(0.0)
    assert trip.rider_access_min == pytest.approx(0.0)
    assert trip.rider_carpool_min == pytest.approx(3.0)
    assert trip.driver_carpool_min == pytest.approx(4.0)
    assert trip.vehicle_min_saved == pytest.approx(2.0)
    rows = {(leg.who, leg.name): leg.edge_rows for leg in trip.legs}
    assert rows["driver", "pickup"] == [0]
    assert rows["driver", "shared"] == [2, 4]
    assert rows["driver", "dropoff"] == [6]


def test_early_arrival_does_not_add_wait(line_network: Network) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster((10, 0.0, 4000.0, 0.0), (11, 1000.0, 3000.0, 10.0))

    trip = direct_itinerary(router, cluster, 10, 11)

    assert trip.rider_wait_min == pytest.approx(0.0)
    assert trip.rider_carpool_min == pytest.approx(trip.shared_min)


def test_a_longer_tour_raises_the_driver_ratio(line_network: Network) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster((10, 0.0, 1000.0, 0.0), (11, 2000.0, 3000.0, 0.0))

    trip = direct_itinerary(router, cluster, 10, 11)

    assert trip.driver_solo_min == pytest.approx(1.0)
    assert trip.driver_tour_min == pytest.approx(5.0)
    assert trip.driver_ratio == pytest.approx(5.0)
    assert trip.vehicle_min_saved == pytest.approx(-3.0)


def test_park_and_ride_wait_falls_on_the_earlier_arrival(line_network: Network) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster((10, 0.0, 4000.0, 0.0), (11, 1000.0, 3000.0, 0.0))

    trip = pnr_itinerary(router, cluster, 10, 11, 0, 2000.0, 0.0)

    assert trip.lot_id == 0
    assert trip.minutes("driver", "pickup") == pytest.approx(2.0)
    assert trip.rider_access_min == pytest.approx(1.0)
    assert trip.shared_min == pytest.approx(1.0)
    assert trip.driver_tour_min == pytest.approx(4.0)
    assert trip.rider_wait_min == pytest.approx(1.0)
    assert trip.driver_wait_min == pytest.approx(0.0)
    assert trip.rider_carpool_min == pytest.approx(3.0)
    assert trip.shared_share == pytest.approx(0.25)
    assert trip.vehicle_min_saved == pytest.approx(1.0)


def test_the_driver_waits_when_the_rider_reaches_the_lot_later(line_network: Network) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster((10, 2000.0, 4000.0, 0.0), (11, 0.0, 3000.0, 0.0))

    trip = pnr_itinerary(router, cluster, 10, 11, 1, 2000.0, 0.0)

    assert trip.driver_wait_min == pytest.approx(2.0)
    assert trip.rider_wait_min == pytest.approx(0.0)
    assert trip.driver_carpool_min == pytest.approx(4.0)
    assert trip.rider_carpool_min == pytest.approx(3.0)
    assert trip.vehicle_min_saved == pytest.approx(1.0)


def test_a_missing_trip_raises(line_network: Network) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster((10, 0.0, 1000.0, 0.0))

    with pytest.raises(ValueError, match="not in the cluster"):
        direct_itinerary(router, cluster, 10, 11)
