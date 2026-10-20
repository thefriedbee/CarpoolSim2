"""Tests for loading a simulation config from JSON or YAML."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from carpoolsim.config import SimulationConfig
from carpoolsim.config import load_config
from carpoolsim.optimizer import BipartiteAssigner
from carpoolsim.optimizer import GreedyAssigner
from carpoolsim.optimizer import MatchingAssigner
from carpoolsim.trip_cluster import TripCluster

ROOT = Path(__file__).resolve().parents[2]


def cluster() -> TripCluster:
    """One trip, enough to build a simulator."""
    zeros = np.zeros(1)
    return TripCluster(zeros, zeros, zeros, zeros, zeros)


def test_defaults_match_an_empty_direct_run() -> None:
    simulator = SimulationConfig().build(cluster())

    assert simulator.horizon == pytest.approx(15.0)
    assert simulator.dt == pytest.approx(5.0)
    assert simulator.et is None
    assert simulator.commit == "window"
    assert isinstance(simulator.assigner, MatchingAssigner)
    assert simulator.modes == (("direct", ()),)


def test_yaml_and_json_load_the_same_settings(tmp_path: Path) -> None:
    data = {
        "horizon": 30,
        "dt": 10,
        "et": 10,
        "commit": "step",
        "assigner": "greedy",
        "modes": [
            {
                "name": "direct",
                "filters": [{"filter": "FilterDepartureTime", "delta1": 10}],
            }
        ],
    }
    yaml_path = tmp_path / "run.yaml"
    yml_path = tmp_path / "run.yml"
    json_path = tmp_path / "run.json"
    yaml_path.write_text(yaml.safe_dump(data))
    yml_path.write_text(yaml.safe_dump(data))
    json_path.write_text(json.dumps(data))

    loaded = load_config(yaml_path)
    assert load_config(yml_path).to_dict() == loaded.to_dict()
    assert load_config(json_path).to_dict() == loaded.to_dict()
    assert loaded.to_dict() == load_config(json_path).to_dict()
    simulator = loaded.build(cluster())
    assert simulator.horizon == pytest.approx(30.0)
    assert simulator.dt == pytest.approx(10.0)
    assert simulator.et == pytest.approx(10.0)
    assert simulator.commit == "step"
    assert isinstance(simulator.assigner, GreedyAssigner)
    assert simulator.modes[0][0] == "direct"
    assert simulator.modes[0][1][0].delta1 == pytest.approx(10.0)


def test_a_config_round_trips_through_the_simulator_it_builds() -> None:
    config = load_config(ROOT / "config" / "gt_2022.yaml")
    built = SimulationConfig.from_simulator(config.build(cluster(), router=object()))

    assert built.to_dict() == config.to_dict()


def test_with_horizon_keeps_the_other_settings() -> None:
    config = load_config(ROOT / "config" / "gt_2022.yaml")
    wider = config.with_horizon(30)

    assert wider.horizon == pytest.approx(30.0)
    assert config.horizon == pytest.approx(15.0)
    assert wider.to_dict()["dt"] == config.to_dict()["dt"]
    assert wider.to_dict()["modes"] == config.to_dict()["modes"]


def test_door_to_door_file_builds_the_georgia_tech_filters() -> None:
    config = load_config(ROOT / "config" / "gt_2022.yaml")
    router = object()
    simulator = config.build(cluster(), router=router)
    filters = simulator.modes[0][1]

    assert [type(step).__name__ for step in filters] == [
        "FilterDepartureTime",
        "FilterOriginEuclideanDistance",
        "FilterRerouteEuclideanDistance",
        "FilterRerouteTravelTime",
        "FilterSharedTravelTime",
        "FilterPassengerWaitTime",
    ]
    assert filters[0].delta1 == pytest.approx(15.0)
    assert filters[1].r1 == pytest.approx(5.0)
    assert filters[2].mu_1 == pytest.approx(1.5)
    assert filters[3].gamma_1 == pytest.approx(1.5)
    assert filters[3].router is router
    assert filters[4].gamma_2 == pytest.approx(0.5)
    assert filters[5].delta2 == pytest.approx(10.0)
    assert isinstance(simulator.assigner, MatchingAssigner)


def test_park_and_ride_file_shares_one_plan() -> None:
    config = load_config(ROOT / "config" / "gt_2022_pnr.yaml")
    router = object()
    lots = (np.array([0.0, 10.0]), np.array([0.0, 0.0]))
    simulator = config.build(cluster(), router=router, lots=lots)
    filters = simulator.modes[0][1]

    assert simulator.modes[0][0] == "pnr"
    assert filters[1].plan is filters[2].plan
    assert filters[1].plan is filters[-1].plan
    assert filters[1].plan.r_p == pytest.approx(5.0)
    assert filters[1].plan.px.tolist() == [0.0, 10.0]
    assert filters[3].mu_3 == pytest.approx(0.5)
    assert filters[4]._travel.router is router
    assert filters[-1].delta2 == pytest.approx(10.0)
    recorded = SimulationConfig.from_simulator(simulator).to_dict()
    assert recorded == config.to_dict()
    assert recorded["modes"][0]["r_p"] == pytest.approx(5.0)
    assert "n_lots" not in recorded["modes"][0]["filters"][1]


def test_bipartite_assigner_is_built_by_name() -> None:
    config = SimulationConfig(assigner="bipartite")

    assert isinstance(config.build(cluster()).assigner, BipartiteAssigner)


def test_clock_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        SimulationConfig(dt=0)
    with pytest.raises(ValueError, match="horizon must be at least dt"):
        SimulationConfig(horizon=5, dt=10)
    with pytest.raises(ValueError, match="et must be at least dt"):
        SimulationConfig(et=4)
    with pytest.raises(ValueError, match="et must be at most horizon"):
        SimulationConfig(et=16)
    with pytest.raises(ValueError, match="commit must be 'window' or 'step'"):
        SimulationConfig(commit="later")
    with pytest.raises(ValueError, match="assigner must be 'matching', 'greedy', or 'bipartite'"):
        SimulationConfig(assigner="exact")
    with pytest.raises(ValueError, match="horizon must be a number"):
        SimulationConfig(horizon=True)


def test_unknown_file_settings_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("horizon: 15\nstep: 5\n")
    with pytest.raises(ValueError, match="unknown config setting 'step'"):
        load_config(path)

    path.write_text("modes:\n  - name: direct\n    reach: 5\n    filters: []\n")
    with pytest.raises(ValueError, match="unknown mode setting 'reach'"):
        load_config(path)

    path.write_text(
        "modes:\n  - name: direct\n    filters:\n      - filter: FilterDepartureTime\n        minutes: 15\n"
    )
    with pytest.raises(ValueError, match="unknown FilterDepartureTime setting 'minutes'"):
        load_config(path)

    path.write_text("modes:\n  - name: direct\n    filters:\n      - filter: FilterNope\n")
    with pytest.raises(ValueError, match="unknown filter 'FilterNope'"):
        load_config(path)

    path.write_text("not a mapping")
    with pytest.raises(ValueError, match="config must be a mapping"):
        load_config(path)

    text = path.with_suffix(".txt")
    text.write_text("{}\n")
    with pytest.raises(ValueError, match="config must be .json, .yaml, or .yml"):
        load_config(text)


def test_a_travel_time_filter_requires_a_router() -> None:
    config = load_config(ROOT / "config" / "gt_2022.yaml")

    with pytest.raises(ValueError, match="FilterRerouteTravelTime requires a router"):
        config.build(cluster())


def test_park_and_ride_requires_r_p_and_lot_coordinates(tmp_path: Path) -> None:
    missing = tmp_path / "pnr.yaml"
    missing.write_text("modes:\n  - name: pnr\n    filters:\n      - filter: FilterPnrAccess\n")
    with pytest.raises(ValueError, match="park-and-ride mode 'pnr' requires r_p"):
        load_config(missing)

    extra = tmp_path / "direct.yaml"
    extra.write_text("modes:\n  - name: direct\n    r_p: 5\n    filters:\n      - filter: FilterDepartureTime\n        delta1: 15\n")
    with pytest.raises(ValueError, match="mode 'direct' has r_p but no park-and-ride filter"):
        load_config(extra)

    config = load_config(ROOT / "config" / "gt_2022_pnr.yaml")
    with pytest.raises(ValueError, match="park-and-ride mode 'pnr' requires lot coordinates"):
        config.build(cluster(), router=object())


def test_a_filter_threshold_is_checked_when_the_simulator_is_built(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        "modes:\n  - name: direct\n    filters:\n      - filter: FilterSharedTravelTime\n        gamma_2: 2\n"
    )
    config = load_config(path)

    with pytest.raises(ValueError, match="gamma_2 must be between 0 and 1"):
        config.build(cluster(), router=object())


def test_mode_names_are_required_once(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("modes: []\n")
    with pytest.raises(ValueError, match="at least one mode is required"):
        load_config(path)

    path.write_text("modes:\n  - filters: []\n")
    with pytest.raises(ValueError, match="mode name must be a non-empty string"):
        load_config(path)

    path.write_text("modes:\n  - name: direct\n    filters: []\n  - name: direct\n    filters: []\n")
    with pytest.raises(ValueError, match="mode names must be unique"):
        load_config(path)
