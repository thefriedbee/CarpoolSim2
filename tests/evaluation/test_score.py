"""Tests for writing a simulation run to tables."""

import json

import pandas as pd
import pytest

from carpoolsim.network_prep import RoadRouter
from carpoolsim.optimizer import GreedyAssigner
from carpoolsim.simulator import Match
from carpoolsim.simulator import Simulator
from carpoolsim.simulator import Solo
from carpoolsim.trip_cluster import FilterDepartureTime
from carpoolsim.trip_cluster import FilterPnrAccess
from carpoolsim.trip_cluster import PnrPlan
from evaluation.score import load_results
from evaluation.score import write_results
from tests.conftest import Network
from tests.evaluation.test_itinerary import line_cluster


def finished(cluster, filters=(), modes=None) -> Simulator:
    """A simulator whose matches and solos the test fills in."""
    simulator = Simulator(
        cluster,
        GreedyAssigner(),
        filters=filters,
        modes=modes,
        horizon=15,
        dt=5,
    )
    simulator.matches = []
    simulator.solos = []
    return simulator


def test_write_results_records_the_pair_and_the_solo(line_network: Network, tmp_path) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster(
        (10, 0.0, 4000.0, 0.0),
        (11, 1000.0, 3000.0, 0.0),
        (12, 0.0, 1000.0, 50.0),
    )
    simulator = finished(cluster, filters=(FilterDepartureTime(15),))
    simulator.matches = [Match(10, 11, -5.0)]
    simulator.solos = [Solo(12, 55.0)]

    tables = write_results(tmp_path, simulator, router, edges)

    driver = tables.travelers.set_index("trip_id").loc[10]
    rider = tables.travelers.set_index("trip_id").loc[11]
    solo = tables.travelers.set_index("trip_id").loc[12]
    assert list(tables.travelers["trip_id"]) == [10, 11, 12]
    assert driver["role"] == "driver"
    assert int(driver["partner_id"]) == 11
    assert driver["delta_min"] == pytest.approx(0.0)
    assert driver["in_vehicle_min"] == pytest.approx(4.0)
    assert rider["role"] == "rider"
    assert rider["wait_min"] == pytest.approx(1.0)
    assert rider["carpool_min"] == pytest.approx(3.0)
    assert rider["delta_min"] == pytest.approx(1.0)
    assert solo["role"] == "solo"
    assert solo["solo_min"] == pytest.approx(1.0)
    assert solo["carpool_min"] == pytest.approx(1.0)
    assert solo["mode"] == ""
    match = tables.matches.iloc[0]
    assert match["vehicle_min_saved"] == pytest.approx(2.0)
    assert match["driver_tour_min"] == pytest.approx(4.0)
    moving = tables.legs[tables.legs["leg"].isin(["pickup", "shared", "dropoff"])]
    assert moving["travel_min"].sum() == pytest.approx(match["driver_tour_min"])
    settings = json.loads((tmp_path / "run.json").read_text())
    assert settings["horizon"] == pytest.approx(15.0)
    assert settings["dt"] == pytest.approx(5.0)
    assert settings["et"] is None
    assert settings["commit"] == "window"
    assert settings["assigner"] == "greedy"
    assert settings["modes"][0]["filters"][0] == {"filter": "FilterDepartureTime", "delta1": 15.0}


def test_leg_lines_meet_at_the_pickup(line_network: Network, tmp_path) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster((10, 0.0, 4000.0, 0.0), (11, 1000.0, 3000.0, 0.0))
    simulator = finished(cluster)
    simulator.matches = [Match(10, 11, 0.0)]

    tables = write_results(tmp_path, simulator, router, edges)

    pickup = tables.legs[(tables.legs["leg"] == "pickup")].geometry.iloc[0]
    shared = tables.legs[(tables.legs["leg"] == "shared")].geometry.iloc[0]
    dropoff = tables.legs[(tables.legs["leg"] == "dropoff")].geometry.iloc[0]
    assert pickup.coords[-1] == pytest.approx(shared.coords[0])
    assert shared.coords[-1] == pytest.approx(dropoff.coords[0])
    assert pickup.coords[0] == pytest.approx((0.0, 0.0))
    assert dropoff.coords[-1] == pytest.approx((4000.0, 0.0))


def test_park_and_ride_keeps_the_lot_the_filters_choose(line_network: Network, tmp_path) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster((10, 0.0, 4000.0, 0.0), (11, 1000.0, 3000.0, 0.0))
    plan = PnrPlan([2000.0], [0.0], 5)
    simulator = finished(cluster, modes=(("pnr", (FilterPnrAccess(plan),)),))
    simulator.matches = [Match(10, 11, 0.0, "pnr")]

    tables = write_results(tmp_path, simulator, router, edges)

    match = tables.matches.iloc[0]
    assert int(match["lot_id"]) == 0
    assert match["lot_x"] == pytest.approx(2000.0)
    assert match["lot_y"] == pytest.approx(0.0)
    assert match["rider_access_min"] == pytest.approx(1.0)
    assert match["vehicle_min_saved"] == pytest.approx(1.0)
    assert "access" in set(tables.legs["leg"])
    rider = tables.travelers.set_index("trip_id").loc[11]
    assert rider["access_min"] == pytest.approx(1.0)
    assert rider["wait_min"] == pytest.approx(1.0)
    assert int(rider["lot_id"]) == 0
    settings = json.loads((tmp_path / "run.json").read_text())
    assert settings["modes"][0]["name"] == "pnr"
    assert settings["modes"][0]["r_p"] == pytest.approx(5.0)
    assert settings["modes"][0]["filters"] == [{"filter": "FilterPnrAccess"}]


def test_tied_lots_keep_the_lower_index(line_network: Network, tmp_path) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster((1, 0.0, 5000.0, 0.0), (2, 4000.0, 5000.0, 0.0))
    plan = PnrPlan([1000.0, 3000.0], [0.0, 0.0], 3)
    simulator = finished(cluster, modes=(("pnr", (FilterPnrAccess(plan),)),))
    simulator.matches = [Match(1, 2, 0.0, "pnr")]

    tables = write_results(tmp_path, simulator, router, edges)

    assert int(tables.matches.iloc[0]["lot_id"]) == 0
    assert tables.matches.iloc[0]["lot_x"] == pytest.approx(1000.0)


def test_load_results_reads_the_written_files(line_network: Network, tmp_path) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster((10, 0.0, 4000.0, 0.0), (11, 1000.0, 3000.0, 0.0))
    simulator = finished(cluster)
    simulator.matches = [Match(10, 11, 0.0)]
    write_results(tmp_path, simulator, router, edges)

    loaded = load_results(tmp_path)

    assert loaded.matches.iloc[0]["vehicle_min_saved"] == pytest.approx(2.0)
    assert loaded.travelers["mode"].tolist() == ["", ""]
    moving = loaded.legs[loaded.legs["leg"].isin(["pickup", "shared", "dropoff"])]
    assert moving["travel_min"].sum() == pytest.approx(loaded.matches.iloc[0]["driver_tour_min"])


def test_solo_path_roundtrip_keeps_the_network_detour(shortcut_network: Network, tmp_path) -> None:
    nodes, edges = shortcut_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster((42, 0.0, 2000.0, 0.0))
    simulator = finished(cluster)
    simulator.solos = [Solo(42, 5.0)]

    write_results(tmp_path, simulator, router, edges)
    loaded = load_results(tmp_path)

    leg = loaded.legs.iloc[0]
    assert int(leg["trip_id"]) == 42
    assert pd.isna(leg["match_id"])
    assert (leg["who"], leg["leg"]) == ("solo", "solo")
    assert leg["travel_min"] == pytest.approx(2.0)
    assert list(leg.geometry.coords) == [(0.0, 0.0), (1000.0, 1000.0), (2000.0, 0.0)]
    assert str(loaded.legs["match_id"].dtype) == "Int64"
    assert str(loaded.legs["trip_id"].dtype) == "Int64"


def test_trip_ids_isolate_solo_and_carpool_routes(line_network: Network, tmp_path) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster(
        (10, 0.0, 4000.0, 0.0),
        (11, 1000.0, 3000.0, 0.0),
        (12, 4000.0, 5000.0, 0.0),
    )
    simulator = finished(cluster)
    simulator.matches = [Match(10, 11, 0.0)]
    simulator.solos = [Solo(12, 5.0)]

    write_results(tmp_path, simulator, router, edges)
    loaded = load_results(tmp_path)

    solo = loaded.legs[loaded.legs["trip_id"] == 12]
    assert len(solo) == 1
    assert solo["match_id"].isna().all()
    assert list(solo.geometry.iloc[0].coords) == [(4000.0, 0.0), (5000.0, 0.0)]
    pair = loaded.legs[loaded.legs["match_id"] == 0]
    assert len(pair) == 5
    assert pair.loc[pair["who"] == "driver", "trip_id"].tolist() == [10, 10, 10, 10]
    assert pair.loc[pair["who"] == "rider", "trip_id"].tolist() == [11]


@pytest.mark.parametrize("carpool", [True, False])
def test_legacy_legs_without_trip_ids_remain_loadable(
    line_network: Network, tmp_path, carpool: bool
) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster((10, 0.0, 4000.0, 0.0), (11, 1000.0, 3000.0, 0.0))
    simulator = finished(cluster)
    if carpool:
        simulator.matches = [Match(10, 11, 0.0)]
    else:
        simulator.solos = [Solo(10, 5.0), Solo(11, 5.0)]
    tables = write_results(tmp_path, simulator, router, edges)
    legacy = tables.legs[tables.legs["match_id"].notna()].drop(columns="trip_id")
    (tmp_path / "legs.gpkg").unlink()
    legacy.to_file(tmp_path / "legs.gpkg", driver="GPKG")

    loaded = load_results(tmp_path)

    assert len(loaded.legs) == (5 if carpool else 0)
    assert str(loaded.legs["trip_id"].dtype) == "Int64"
    if carpool:
        assert loaded.legs.loc[loaded.legs["who"] == "driver", "trip_id"].tolist() == [10] * 4
        assert loaded.legs.loc[loaded.legs["who"] == "rider", "trip_id"].tolist() == [11]


def test_a_day_with_no_carpool_still_writes(line_network: Network, tmp_path) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster((7, 0.0, 1000.0, 30.0))
    simulator = finished(cluster)
    simulator.solos = [Solo(7, 35.0)]

    tables = write_results(tmp_path, simulator, router, edges)
    loaded = load_results(tmp_path)

    assert tables.matches.empty
    assert loaded.matches.empty
    assert len(loaded.legs) == 1
    assert loaded.legs.iloc[0]["trip_id"] == 7
    assert pd.isna(loaded.legs.iloc[0]["match_id"])
    assert loaded.travelers.iloc[0]["role"] == "solo"
    assert loaded.travelers.iloc[0]["solo_min"] == pytest.approx(1.0)


def test_an_empty_run_roundtrips(line_network: Network, tmp_path) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    simulator = finished(line_cluster())

    tables = write_results(tmp_path, simulator, router, edges)
    loaded = load_results(tmp_path)

    assert tables.travelers.empty
    assert loaded.travelers.empty
    assert loaded.matches.empty
    assert loaded.legs.empty
    assert list(loaded.legs.columns) == [
        "match_id", "trip_id", "who", "leg", "travel_min", "geometry"
    ]
    assert str(loaded.legs["match_id"].dtype) == "Int64"
    assert str(loaded.legs["trip_id"].dtype) == "Int64"


def test_every_traveler_must_be_recorded(line_network: Network, tmp_path) -> None:
    nodes, edges = line_network
    router = RoadRouter(nodes, edges)
    cluster = line_cluster((10, 0.0, 1000.0, 0.0), (11, 1000.0, 2000.0, 0.0))
    simulator = finished(cluster)
    simulator.matches = [Match(10, 11, 0.0), Match(10, 11, 5.0)]

    with pytest.raises(ValueError, match="more than once"):
        write_results(tmp_path, simulator, router, edges)

    simulator.matches = [Match(10, 11, 0.0, "missing")]
    with pytest.raises(ValueError, match="not a simulator mode"):
        write_results(tmp_path, simulator, router, edges)
