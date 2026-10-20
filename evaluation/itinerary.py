"""Travel times and paths for drive-alone trips and committed carpools."""

from __future__ import annotations

from pyproj import Transformer

from carpoolsim.network_prep.router import RoadRouter
from carpoolsim.network_prep.router import Route
from carpoolsim.trip_cluster.cluster import TripCluster

_TO_LONLAT = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


class Leg:
    """One drive in a carpool or in the solo trip it replaces.

    ``who`` is ``driver``, ``rider``, or ``solo``. ``name`` is ``solo``, ``pickup``,
    ``shared``, ``dropoff``, or ``access``. ``travel_min`` is the free-flow
    time in minutes. ``edge_rows`` are positions in the edge table the router
    was built from, in path order. ``start`` and ``end`` are the EPSG:3857
    waypoints of that leg.
    """

    def __init__(
        self,
        who: str,
        name: str,
        travel_min: float,
        edge_rows: list[int],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        self.who = who
        self.name = name
        self.travel_min = travel_min
        self.edge_rows = edge_rows
        self.start = start
        self.end = end


class Itinerary:
    """Legs of one pair, and the wait at the meeting point.

    Door-to-door time includes waiting. The driver's tour is the moving
    time only: pickup, shared ride, and dropoff. ``vehicle_min_saved``
    compares that moving time, plus the rider's own drive to a lot, with
    the two solo drives.

    On a direct trip the driver does not wait. A rider whose driver arrives
    before their departure also waits zero and leaves early. At a
    park-and-ride lot the person who arrives first waits for the other.
    ``lot_id`` is that lot, or None on a direct trip.
    """

    def __init__(
        self,
        legs: list[Leg],
        driver_wait_min: float,
        rider_wait_min: float,
        lot_id: int | None = None,
    ) -> None:
        self.legs = legs
        self.driver_wait_min = driver_wait_min
        self.rider_wait_min = rider_wait_min
        self.lot_id = lot_id

    def minutes(self, who: str, name: str) -> float:
        """Free-flow minutes of the leg named ``name`` for ``who``."""
        return self._leg(who, name).travel_min

    @property
    def driver_solo_min(self) -> float:
        """Driver's origin-to-destination time alone."""
        return self.minutes("driver", "solo")

    @property
    def rider_solo_min(self) -> float:
        """Rider's origin-to-destination time alone."""
        return self.minutes("rider", "solo")

    @property
    def driver_tour_min(self) -> float:
        """Moving time of the driver's carpool tour, in minutes."""
        return (
            self.minutes("driver", "pickup")
            + self.minutes("driver", "shared")
            + self.minutes("driver", "dropoff")
        )

    @property
    def shared_min(self) -> float:
        """Minutes the rider spends in the driver's car."""
        return self.minutes("driver", "shared")

    @property
    def rider_access_min(self) -> float:
        """Rider's own drive to the lot. Zero on a direct trip."""
        if self.lot_id is None:
            return 0.0
        return self.minutes("rider", "access")

    @property
    def driver_carpool_min(self) -> float:
        """Driver's door-to-door time, including any wait at a lot."""
        return self.driver_tour_min + self.driver_wait_min

    @property
    def rider_carpool_min(self) -> float:
        """Rider's door-to-door time, including access and waiting."""
        return self.rider_access_min + self.rider_wait_min + self.shared_min

    @property
    def driver_ratio(self) -> float:
        """Driver's tour divided by their solo trip."""
        return _ratio(self.driver_tour_min, self.driver_solo_min, "driver solo travel time")

    @property
    def shared_share(self) -> float:
        """Shared leg divided by the driver's tour."""
        return _ratio(self.shared_min, self.driver_tour_min, "driver tour")

    @property
    def vehicle_min_saved(self) -> float:
        """Solo driving minus the carpool's moving time, in minutes."""
        return (
            self.driver_solo_min
            + self.rider_solo_min
            - self.driver_tour_min
            - self.rider_access_min
        )

    def _leg(self, who: str, name: str) -> Leg:
        for leg in self.legs:
            if leg.who == who and leg.name == name:
                return leg
        raise KeyError(f"no {who} {name} leg")


def solo_leg(router: RoadRouter, cluster: TripCluster, traveler: int) -> Leg:
    """The routed drive from this traveler's origin to their destination.

    The returned leg has ``who='solo'`` and ``name='solo'``. Its edge rows
    refer to the edge table used to build ``router``, like carpool legs.
    """
    routes = _Routes(router)
    stop = _stop(cluster, traveler)
    origin = routes.snap(stop[0], stop[1])
    destination = routes.snap(stop[2], stop[3])
    return _leg(routes, "solo", "solo", origin, destination, stop[0:2], stop[2:4])


def solo_minutes(router: RoadRouter, cluster: TripCluster, traveler: int) -> float:
    """Free-flow minutes from this traveler's origin to their destination."""
    return solo_leg(router, cluster, traveler).travel_min


def direct_itinerary(router: RoadRouter, cluster: TripCluster, driver: int, rider: int) -> Itinerary:
    """Build the pickup, shared ride, and dropoff for a direct carpool.

    The driver goes from their origin to the rider's origin, then to the
    rider's destination, then to their own destination. The rider is picked
    up at home and dropped off at their own destination.
    """
    routes = _Routes(router)
    driver_stop = _stop(cluster, driver)
    rider_stop = _stop(cluster, rider)
    origin_driver = routes.snap(driver_stop[0], driver_stop[1])
    dest_driver = routes.snap(driver_stop[2], driver_stop[3])
    origin_rider = routes.snap(rider_stop[0], rider_stop[1])
    dest_rider = routes.snap(rider_stop[2], rider_stop[3])
    legs = [
        _leg(routes, "driver", "solo", origin_driver, dest_driver, driver_stop[0:2], driver_stop[2:4]),
        _leg(routes, "rider", "solo", origin_rider, dest_rider, rider_stop[0:2], rider_stop[2:4]),
        _leg(routes, "driver", "pickup", origin_driver, origin_rider, driver_stop[0:2], rider_stop[0:2]),
        _leg(routes, "driver", "shared", origin_rider, dest_rider, rider_stop[0:2], rider_stop[2:4]),
        _leg(routes, "driver", "dropoff", dest_rider, dest_driver, rider_stop[2:4], driver_stop[2:4]),
    ]
    rider_wait = max(0.0, driver_stop[4] + legs[2].travel_min - rider_stop[4])
    return Itinerary(legs, 0.0, rider_wait, None)


def pnr_itinerary(
    router: RoadRouter,
    cluster: TripCluster,
    driver: int,
    rider: int,
    lot_id: int,
    lot_x: float,
    lot_y: float,
) -> Itinerary:
    """Build the lot meeting, shared ride, and dropoff for a park-and-ride pair.

    The driver goes from their origin to the lot, then to the rider's
    destination, then to their own destination. The rider drives from home
    to the same lot and rides from there.
    """
    routes = _Routes(router)
    driver_stop = _stop(cluster, driver)
    rider_stop = _stop(cluster, rider)
    origin_driver = routes.snap(driver_stop[0], driver_stop[1])
    dest_driver = routes.snap(driver_stop[2], driver_stop[3])
    origin_rider = routes.snap(rider_stop[0], rider_stop[1])
    dest_rider = routes.snap(rider_stop[2], rider_stop[3])
    park = routes.snap(lot_x, lot_y)
    lot = (float(lot_x), float(lot_y))
    legs = [
        _leg(routes, "driver", "solo", origin_driver, dest_driver, driver_stop[0:2], driver_stop[2:4]),
        _leg(routes, "rider", "solo", origin_rider, dest_rider, rider_stop[0:2], rider_stop[2:4]),
        _leg(routes, "driver", "pickup", origin_driver, park, driver_stop[0:2], lot),
        _leg(routes, "rider", "access", origin_rider, park, rider_stop[0:2], lot),
        _leg(routes, "driver", "shared", park, dest_rider, lot, rider_stop[2:4]),
        _leg(routes, "driver", "dropoff", dest_rider, dest_driver, rider_stop[2:4], driver_stop[2:4]),
    ]
    rider_wait, driver_wait = _lot_wait(
        driver_stop[4], legs[2].travel_min, rider_stop[4], legs[3].travel_min
    )
    return Itinerary(legs, driver_wait, rider_wait, int(lot_id))


class _Routes:
    """Snap points and remember routes already computed for one pair."""

    def __init__(self, router: RoadRouter) -> None:
        self.router = router
        self._routes: dict[tuple[int, int], Route] = {}

    def snap(self, x: float, y: float) -> int:
        """Return the drive-network node nearest to an EPSG:3857 point."""
        lon, lat = _TO_LONLAT.transform(x, y)
        return self.router.nearest_node(lon, lat)

    def between(self, origin: int, destination: int) -> Route:
        """Return the shortest path, using a path already computed in this pair."""
        key = (origin, destination)
        route = self._routes.get(key)
        if route is None:
            route = self.router.route(origin, destination)
            self._routes[key] = route
        return route


def _stop(cluster: TripCluster, trip_id: int) -> tuple[float, float, float, float, float]:
    """Origin, destination, and departure for ``trip_id``."""
    position = {int(traveler): index for index, traveler in enumerate(cluster.ids)}
    try:
        index = position[int(trip_id)]
    except KeyError as error:
        raise ValueError(f"trip {trip_id} is not in the cluster") from error
    return (
        float(cluster.ox[index]),
        float(cluster.oy[index]),
        float(cluster.dx[index]),
        float(cluster.dy[index]),
        float(cluster.t[index]),
    )


def _leg(
    routes: _Routes,
    who: str,
    name: str,
    origin: int,
    destination: int,
    start: tuple[float, float],
    end: tuple[float, float],
) -> Leg:
    """Route one leg, or report which leg has no path."""
    try:
        route = routes.between(origin, destination)
    except RuntimeError as error:
        raise RuntimeError(f"no path for the {who} {name} leg") from error
    return Leg(who, name, route.travel_time, list(route.edge_rows), start, end)


def _lot_wait(
    driver_departure: float,
    driver_drive: float,
    rider_departure: float,
    rider_drive: float,
) -> tuple[float, float]:
    """Return the rider's wait and the driver's wait at the lot.

    Each person arrives at their departure plus the drive to the lot. The
    earlier arrival waits. The later arrival waits zero.
    """
    gap = (driver_departure + driver_drive) - (rider_departure + rider_drive)
    if gap >= 0:
        return gap, 0.0
    return 0.0, -gap


def _ratio(part: float, whole: float, label: str) -> float:
    """Return ``part / whole`` when ``whole`` is positive."""
    if not whole > 0:
        raise ValueError(f"{label} must be positive")
    return part / whole
