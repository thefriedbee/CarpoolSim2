"""Write the travelers, carpools, and paths of one simulation run."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge

from carpoolsim.config import SimulationConfig
from carpoolsim.network_prep.router import RoadRouter
from carpoolsim.simulator.simulator import Simulator
from carpoolsim.trip_cluster.cluster import TripCluster
from carpoolsim.trip_cluster.filters.pnr.plan import PnrPlan
from carpoolsim.trip_cluster.share import Shareability
from evaluation.itinerary import Itinerary
from evaluation.itinerary import Leg
from evaluation.itinerary import direct_itinerary
from evaluation.itinerary import pnr_itinerary
from evaluation.itinerary import solo_leg


class RunTables:
    """Traveler rows, carpool rows, and leg lines for one run."""

    def __init__(
        self,
        travelers: pd.DataFrame,
        matches: pd.DataFrame,
        legs: gpd.GeoDataFrame,
    ) -> None:
        self.travelers = travelers
        self.matches = matches
        self.legs = legs


def write_results(
    directory: str | Path,
    simulator: Simulator,
    router: RoadRouter,
    edges: gpd.GeoDataFrame,
) -> RunTables:
    """Score the committed pairs and write them under ``directory``.

    ``edges`` is the edge table the router was built from. Leg geometry uses
    positions in that table. A park-and-ride mode is any mode whose filters
    hold a ``PnrPlan``. Scoring replays those filters on the committed pair
    so the lot is the one that mode would choose, and that replay replaces
    ``plan.lot``.

    The files are ``run.json``, ``travelers.csv``, ``matches.csv``, and
    ``legs.gpkg``. ``run.json`` is the simulation config: the clock, the
    assigner, and each mode's filters. A park-and-ride mode records ``r_p``
    once. Door-to-door columns include waiting. ``driver_tour_min`` and
    ``vehicle_min_saved`` count moving time only. Each leg records its
    traveler's ``trip_id``. Drive-alone legs have ``who='solo'``,
    ``leg='solo'``, and no ``match_id``.
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    _require_coverage(simulator)
    modes = {name: filters for name, filters in simulator.modes}
    traveler_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    leg_rows: list[dict[str, object]] = []
    by_id = _positions(simulator.cluster)

    for match_id, match in enumerate(simulator.matches):
        try:
            filters = modes[match.mode]
        except KeyError as error:
            raise ValueError(f"match mode {match.mode!r} is not a simulator mode") from error
        plan = _plan_of(filters)
        if plan is None:
            trip = direct_itinerary(router, simulator.cluster, match.driver, match.rider)
            lot_x = None
            lot_y = None
        else:
            lot_id = _chosen_lot(simulator.cluster, match.driver, match.rider, filters, plan)
            lot_x = float(plan.px[lot_id])
            lot_y = float(plan.py[lot_id])
            trip = pnr_itinerary(
                router,
                simulator.cluster,
                match.driver,
                match.rider,
                lot_id,
                lot_x,
                lot_y,
            )
        match_rows.append(
            _match_row(match_id, match.driver, match.rider, match.mode, match.t, trip, lot_x, lot_y)
        )
        traveler_rows.extend(
            _pair_rows(simulator.cluster, by_id, match.driver, match.rider, match.mode, match.t, trip)
        )
        leg_rows.extend(_leg_rows(match_id, match.driver, match.rider, trip, edges))

    for solo in simulator.solos:
        leg = solo_leg(router, simulator.cluster, solo.traveler)
        traveler_rows.append(
            _solo_row(simulator.cluster, by_id, solo.traveler, solo.t, leg.travel_min)
        )
        leg_rows.append(_leg_row(None, solo.traveler, leg, edges))

    tables = RunTables(
        _as_travelers(traveler_rows, simulator.cluster),
        _as_matches(match_rows),
        _as_legs(leg_rows),
    )
    _write(folder, simulator, tables)
    return tables


def load_results(directory: str | Path) -> RunTables:
    """Read current or older run tables.

    Older leg files contain only carpool legs and omit ``trip_id``. For
    those files the traveler is recovered from the match and ``who``;
    drive-alone route geometry remains absent.
    """
    folder = Path(directory)
    travelers = pd.read_csv(folder / "travelers.csv")
    matches = pd.read_csv(folder / "matches.csv")
    travelers["mode"] = travelers["mode"].fillna("")
    matches["mode"] = matches["mode"].fillna("")
    for frame, columns in (
        (travelers, ("partner_id", "lot_id")),
        (matches, ("lot_id",)),
    ):
        for column in columns:
            frame[column] = pd.array(frame[column], dtype="Int64")
    legs = gpd.read_file(folder / "legs.gpkg")
    if "trip_id" not in legs:
        paired = matches.set_index("match_id")
        trip_ids = pd.Series(pd.NA, index=legs.index, dtype="Int64")
        for who, column in (("driver", "driver_id"), ("rider", "rider_id")):
            mask = legs["who"] == who
            trip_ids.loc[mask] = legs.loc[mask, "match_id"].map(paired[column])
        legs["trip_id"] = trip_ids
    for column in ("match_id", "trip_id"):
        legs[column] = pd.array(legs[column], dtype="Int64")
    return RunTables(travelers, matches, legs)


def _require_coverage(simulator: Simulator) -> None:
    """Raise unless every traveler is matched or solo exactly once."""
    seen: list[int] = []
    for match in simulator.matches:
        seen.append(int(match.driver))
        seen.append(int(match.rider))
    for solo in simulator.solos:
        seen.append(int(solo.traveler))
    if len(seen) != len(set(seen)):
        raise ValueError("a traveler is matched or solo more than once")
    if set(seen) != {int(trip_id) for trip_id in simulator.cluster.ids}:
        raise ValueError("every traveler must be matched or solo")


def _positions(cluster: TripCluster) -> dict[int, int]:
    """Map a traveler id to its row in ``cluster``."""
    return {int(trip_id): index for index, trip_id in enumerate(cluster.ids)}


def _plan_of(filters: Sequence[object]) -> PnrPlan | None:
    """Return the park-and-ride plan stored on these filters, if they have one."""
    for step in filters:
        plan = getattr(step, "plan", None)
        if isinstance(plan, PnrPlan):
            return plan
    return None


def _chosen_lot(
    cluster: TripCluster,
    driver: int,
    rider: int,
    filters: Sequence[object],
    plan: PnrPlan,
) -> int:
    """Replay the mode on this pair and return the lot it stores.

    Row 0 drives and row 1 rides, which is the direction the simulator committed.
    """
    pair = _two_person_cluster(cluster, driver, rider)
    share = Shareability(pair)
    share.apply(*filters)
    try:
        chosen = int(plan.require(2)[0, 1])
    except RuntimeError as error:
        raise ValueError("committed park-and-ride pair has no lot") from error
    if chosen < 0:
        raise ValueError("committed park-and-ride pair has no lot")
    return chosen


def _two_person_cluster(cluster: TripCluster, driver: int, rider: int) -> TripCluster:
    """Copy the driver and the rider, in that order, keeping their ids."""
    position = _positions(cluster)
    index = [position[int(driver)], position[int(rider)]]
    return TripCluster(
        cluster.ox[index],
        cluster.oy[index],
        cluster.dx[index],
        cluster.dy[index],
        cluster.t[index],
        cluster.ids[index],
    )


def _pair_rows(
    cluster: TripCluster,
    position: dict[int, int],
    driver: int,
    rider: int,
    mode: str,
    commit_min: float,
    trip: Itinerary,
) -> list[dict[str, object]]:
    """One traveler row for the driver and one for the rider."""
    return [
        _traveler_row(
            cluster,
            position[driver],
            role="driver",
            partner_id=rider,
            mode=mode,
            commit_min=commit_min,
            solo_min=trip.driver_solo_min,
            carpool_min=trip.driver_carpool_min,
            wait_min=trip.driver_wait_min,
            access_min=0.0,
            in_vehicle_min=trip.driver_tour_min,
            lot_id=trip.lot_id,
        ),
        _traveler_row(
            cluster,
            position[rider],
            role="rider",
            partner_id=driver,
            mode=mode,
            commit_min=commit_min,
            solo_min=trip.rider_solo_min,
            carpool_min=trip.rider_carpool_min,
            wait_min=trip.rider_wait_min,
            access_min=trip.rider_access_min,
            in_vehicle_min=trip.shared_min,
            lot_id=trip.lot_id,
        ),
    ]


def _solo_row(
    cluster: TripCluster,
    position: dict[int, int],
    traveler: int,
    commit_min: float,
    solo_min: float,
) -> dict[str, object]:
    """One traveler who drove alone, timed on their own origin-to-destination path."""
    index = position[int(traveler)]
    return _traveler_row(
        cluster,
        index,
        role="solo",
        partner_id=None,
        mode="",
        commit_min=commit_min,
        solo_min=solo_min,
        carpool_min=solo_min,
        wait_min=0.0,
        access_min=0.0,
        in_vehicle_min=solo_min,
        lot_id=None,
    )


def _traveler_row(
    cluster: TripCluster,
    index: int,
    role: str,
    partner_id: int | None,
    mode: str,
    commit_min: float,
    solo_min: float,
    carpool_min: float,
    wait_min: float,
    access_min: float,
    in_vehicle_min: float,
    lot_id: int | None,
) -> dict[str, object]:
    """One person, with their solo time and their carpool time."""
    return {
        "trip_id": int(cluster.ids[index]),
        "role": role,
        "partner_id": partner_id,
        "mode": mode,
        "departure_min": float(cluster.t[index]),
        "commit_min": float(commit_min),
        "origin_x": float(cluster.ox[index]),
        "origin_y": float(cluster.oy[index]),
        "dest_x": float(cluster.dx[index]),
        "dest_y": float(cluster.dy[index]),
        "solo_min": solo_min,
        "carpool_min": carpool_min,
        "delta_min": carpool_min - solo_min,
        "wait_min": wait_min,
        "access_min": access_min,
        "in_vehicle_min": in_vehicle_min,
        "lot_id": lot_id,
    }


def _match_row(
    match_id: int,
    driver: int,
    rider: int,
    mode: str,
    commit_min: float,
    trip: Itinerary,
    lot_x: float | None,
    lot_y: float | None,
) -> dict[str, object]:
    """One car, with the tour, the rider's door-to-door time, and the minutes saved."""
    return {
        "match_id": match_id,
        "driver_id": int(driver),
        "rider_id": int(rider),
        "mode": mode,
        "commit_min": float(commit_min),
        "lot_id": trip.lot_id,
        "lot_x": lot_x,
        "lot_y": lot_y,
        "driver_solo_min": trip.driver_solo_min,
        "driver_tour_min": trip.driver_tour_min,
        "driver_ratio": trip.driver_ratio,
        "shared_min": trip.shared_min,
        "shared_share": trip.shared_share,
        "rider_solo_min": trip.rider_solo_min,
        "rider_door_min": trip.rider_carpool_min,
        "rider_wait_min": trip.rider_wait_min,
        "rider_access_min": trip.rider_access_min,
        "vehicle_min_saved": trip.vehicle_min_saved,
    }


def _leg_rows(
    match_id: int,
    driver: int,
    rider: int,
    trip: Itinerary,
    edges: gpd.GeoDataFrame,
) -> list[dict[str, object]]:
    """One GeoPackage row per leg, with the network line when the path has edges."""
    travelers = {"driver": driver, "rider": rider}
    return [_leg_row(match_id, travelers[leg.who], leg, edges) for leg in trip.legs]


def _leg_row(
    match_id: int | None,
    trip_id: int,
    leg: Leg,
    edges: gpd.GeoDataFrame,
) -> dict[str, object]:
    """A routed leg belonging to one traveler, optionally within a carpool."""
    return {
        "match_id": match_id,
        "trip_id": int(trip_id),
        "who": leg.who,
        "leg": leg.name,
        "travel_min": leg.travel_min,
        "geometry": _line(edges, leg.edge_rows, leg.start),
    }


def _line(edges: gpd.GeoDataFrame, rows: list[int], start: tuple[float, float]) -> BaseGeometry:
    """Join the edge lines in path order.

    A path with no edges is a point at ``start``. The joined line starts at
    the first edge's start.
    """
    if not rows:
        return LineString([start, start])
    merged = linemerge([edges.geometry.iloc[row] for row in rows])
    if merged.geom_type != "LineString":
        return merged
    first = edges.geometry.iloc[rows[0]].coords[0]
    if abs(merged.coords[0][0] - first[0]) + abs(merged.coords[0][1] - first[1]) > 1.0:
        return LineString(list(merged.coords)[::-1])
    return merged


def _as_travelers(rows: list[dict[str, object]], cluster: TripCluster) -> pd.DataFrame:
    """Travelers in cluster order, with empty partner and lot ids as integers."""
    columns = [
        "trip_id",
        "role",
        "partner_id",
        "mode",
        "departure_min",
        "commit_min",
        "origin_x",
        "origin_y",
        "dest_x",
        "dest_y",
        "solo_min",
        "carpool_min",
        "delta_min",
        "wait_min",
        "access_min",
        "in_vehicle_min",
        "lot_id",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        frame["partner_id"] = pd.Series(dtype="Int64")
        frame["lot_id"] = pd.Series(dtype="Int64")
        return frame
    order = {int(trip_id): index for index, trip_id in enumerate(cluster.ids)}
    frame = frame.sort_values("trip_id", key=lambda ids: ids.map(order)).reset_index(drop=True)
    frame["partner_id"] = frame["partner_id"].astype("Int64")
    frame["lot_id"] = frame["lot_id"].astype("Int64")
    return frame


def _as_matches(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Carpool rows, including the columns when the run has no carpools."""
    columns = [
        "match_id",
        "driver_id",
        "rider_id",
        "mode",
        "commit_min",
        "lot_id",
        "lot_x",
        "lot_y",
        "driver_solo_min",
        "driver_tour_min",
        "driver_ratio",
        "shared_min",
        "shared_share",
        "rider_solo_min",
        "rider_door_min",
        "rider_wait_min",
        "rider_access_min",
        "vehicle_min_saved",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame["lot_id"] = frame["lot_id"].astype("Int64")
    return frame


def _as_legs(rows: list[dict[str, object]]) -> gpd.GeoDataFrame:
    """Leg lines in EPSG:3857 with nullable match and traveler identifiers."""
    columns = ["match_id", "trip_id", "who", "leg", "travel_min", "geometry"]
    frame = gpd.GeoDataFrame(rows, columns=columns, geometry="geometry", crs="EPSG:3857")
    frame["match_id"] = pd.array(frame["match_id"], dtype="Int64")
    frame["trip_id"] = pd.array(frame["trip_id"], dtype="Int64")
    return frame


def _write(folder: Path, simulator: Simulator, tables: RunTables) -> None:
    """Replace the four result files in ``folder``."""
    settings = SimulationConfig.from_simulator(simulator).to_dict()
    (folder / "run.json").write_text(json.dumps(settings, indent=2) + "\n")
    tables.travelers.to_csv(folder / "travelers.csv", index=False)
    tables.matches.to_csv(folder / "matches.csv", index=False)
    legs_path = folder / "legs.gpkg"
    if legs_path.exists():
        legs_path.unlink()
    tables.legs.to_file(legs_path, driver="GPKG")
