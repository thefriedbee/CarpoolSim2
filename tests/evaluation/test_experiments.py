"""Independent parameter changes and reproducible simulation speed records."""

import copy
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from carpoolsim.config import SimulationConfig
from carpoolsim.config import load_config
from carpoolsim.network_prep import RoadRouter
from carpoolsim.simulator import Match
from carpoolsim.simulator import Solo
from carpoolsim.trip_cluster import TripCluster
from evaluation.experiments import collect_experiment_metrics
from evaluation.experiments import experiment_specs
from evaluation.experiments import MemoizedRoadRouter
from evaluation.experiments import write_experiment_metrics
from evaluation.score import RunTables

ROOT = Path(__file__).resolve().parents[2]


def test_memoized_snapping_is_exact_and_starts_empty_per_router(line_network, monkeypatch) -> None:
    nodes, edges = line_network
    points = [(0.0, 0.0), (0.018, 0.0), (0.045, 0.0), (0.01800000001, 0.0)]
    plain = RoadRouter(nodes, edges)
    expected = [plain.nearest_node(*point) for point in points]
    calls = []
    original = RoadRouter.nearest_node

    def counted(router, lon, lat):
        calls.append((router, lon, lat))
        return original(router, lon, lat)

    monkeypatch.setattr(RoadRouter, "nearest_node", counted)
    first = MemoizedRoadRouter(nodes, edges)
    second = MemoizedRoadRouter(nodes, edges)
    assert calls == []
    assert first._nearest_node_cache == {}
    assert second._nearest_node_cache == {}
    for _repeat in range(3):
        assert [first.nearest_node(*point) for point in points] == expected
    assert len(calls) == len(points)
    assert second._nearest_node_cache == {}
    assert second.nearest_node(*points[0]) == expected[0]
    assert len(calls) == len(points) + 1
    assert calls[-1][0] is second


@pytest.mark.parametrize("cached", [False, True])
def test_snap_memoization_preserves_routes(line_network, cached) -> None:
    nodes, edges = line_network
    plain = RoadRouter(nodes, edges, cached=cached, max_nodes=3)
    memoized = MemoizedRoadRouter(nodes, edges, cached=cached, max_nodes=3)
    for start, end in ((0, 5), (5, 0), (2, 2)):
        expected = plain.route(start, end)
        actual = memoized.route(start, end)
        assert actual.travel_time == expected.travel_time
        assert actual.nodes == expected.nodes
        assert actual.edge_rows == expected.edge_rows
    assert memoized._nearest_node_cache == {}


def leaves(value, prefix="") -> dict:
    """Flatten dictionaries and lists to detect every changed config leaf."""
    if isinstance(value, dict):
        return {
            leaf: item
            for key, child in value.items()
            for leaf, item in leaves(child, f"{prefix}.{key}" if prefix else key).items()
        }
    if isinstance(value, list):
        return {
            leaf: item
            for index, child in enumerate(value)
            for leaf, item in leaves(child, f"{prefix}.{index}").items()
        }
    return {prefix: value}


def test_plan_has_valid_independent_configs() -> None:
    specs = experiment_specs(ROOT)

    assert len(specs) == 60
    assert len({spec["experiment_id"] for spec in specs}) == 60
    assert [spec["mode"] for spec in specs] == (
        ["direct"] * 10 + ["pnr"] * 10 + ["both"] + ["direct"] * 18 + ["pnr"] * 21
    )
    assert specs[20]["experiment_id"] == "e21_both_direct_first"
    assert specs[21]["experiment_id"] == "e22_direct_dt_1"
    assert specs[-1]["experiment_id"] == "e60_pnr_delta2_20"
    by_id = {spec["experiment_id"]: spec for spec in specs}
    direct = load_config(ROOT / "config" / "gt_2022.yaml").to_dict()
    pnr = load_config(ROOT / "config" / "gt_2022_pnr.yaml").to_dict()
    for index, spec in enumerate(specs, start=1):
        assert spec["experiment_id"].startswith(f"e{index:02d}_")
        assert SimulationConfig.from_dict(spec["config"]).to_dict() == spec["config"]
        baseline = by_id[spec["baseline_id"]]
        baseline_leaves = leaves(baseline["config"])
        changed_leaves = leaves(spec["config"])
        differences = [key for key in baseline_leaves if baseline_leaves[key] != changed_leaves[key]]
        if spec["parameter"] == "baseline":
            assert differences == []
            if spec["mode"] == "direct":
                assert spec["config"] == direct
            elif spec["mode"] == "pnr":
                assert spec["config"] == pnr
            else:
                assert spec["mode"] == "both"
                assert spec["config"]["modes"] == [direct["modes"][0], pnr["modes"][0]]
                for key in ("horizon", "dt", "et", "commit", "assigner"):
                    assert spec["config"][key] == direct[key] == pnr[key]
            assert spec["baseline_value"] is None
            assert spec["value"] is None
        else:
            assert differences == [spec["parameter"]]
            assert baseline_leaves[spec["parameter"]] == spec["baseline_value"]
            assert changed_leaves[spec["parameter"]] == spec["value"]


def test_plan_covers_the_clock_and_each_numeric_filter() -> None:
    specs = experiment_specs(ROOT)
    for mode in ("direct", "pnr"):
        group = [spec for spec in specs if spec["mode"] == mode]
        numeric_controls = {
            key for key, value in leaves(group[0]["config"]).items()
            if isinstance(value, int | float)
        }
        varied = {spec["parameter"] for spec in group[1:]}
        assert numeric_controls <= varied
        assert len(varied) == 9

    original = copy.deepcopy(specs[0]["config"])
    specs[1]["config"]["modes"][0]["filters"][0]["delta1"] = -123
    assert specs[0]["config"] == original
    assert experiment_specs(ROOT)[0]["config"] == original


def _levels(specs: list[dict], mode: str, parameter: str) -> set[float]:
    baseline = next(spec for spec in specs if spec["mode"] == mode and spec["parameter"] == "baseline")
    values = [leaves(baseline["config"])[parameter]]
    values.extend(
        spec["value"] for spec in specs if spec["mode"] == mode and spec["parameter"] == parameter
    )
    return {float(value) for value in values}


def test_sensitive_controls_have_five_levels_and_flat_controls_stay_at_two() -> None:
    specs = experiment_specs(ROOT)
    five = {
        ("direct", "dt"): {1, 3, 5, 10, 15},
        ("direct", "modes.0.filters.0.delta1"): {5, 10, 15, 20, 30},
        ("direct", "modes.0.filters.1.r1"): {1, 2.5, 5, 10, 15},
        ("direct", "modes.0.filters.3.gamma_1"): {1.1, 1.25, 1.5, 2, 2.5},
        ("direct", "modes.0.filters.4.gamma_2"): {0.25, 0.5, 0.75, 0.9, 1},
        ("direct", "modes.0.filters.5.delta2"): {0, 5, 10, 15, 20},
        ("pnr", "dt"): {1, 3, 5, 10, 15},
        ("pnr", "modes.0.filters.0.delta1"): {5, 10, 15, 20, 30},
        ("pnr", "modes.0.r_p"): {1, 2.5, 5, 10, 15},
        ("pnr", "modes.0.filters.3.mu_3"): {0.1, 0.25, 0.5, 0.75, 1},
        ("pnr", "modes.0.filters.4.gamma_1"): {1.1, 1.25, 1.5, 2, 2.5},
        ("pnr", "modes.0.filters.5.gamma_2"): {0.25, 0.5, 0.75, 0.9, 1},
        ("pnr", "modes.0.filters.6.delta2"): {0, 5, 10, 15, 20},
    }
    for (mode, parameter), expected in five.items():
        assert _levels(specs, mode, parameter) == expected
    for mode, parameter in (
        ("direct", "horizon"),
        ("direct", "modes.0.filters.2.mu_1"),
        ("pnr", "horizon"),
        ("pnr", "modes.0.filters.2.mu_1"),
    ):
        found = [spec for spec in specs if spec["mode"] == mode and spec["parameter"] == parameter]
        assert len(found) == 1
    assigners = [spec for spec in specs if spec["mode"] == "direct" and spec["parameter"] == "assigner"]
    assert [spec["value"] for spec in assigners] == ["greedy"]


def completed_run(trips=3, matched=True):
    config = SimulationConfig()
    cluster = TripCluster([0] * trips, [0] * trips, [1] * trips, [0] * trips, [0] * trips)
    simulator = config.build(cluster)
    paired = 2 if matched and trips >= 2 else 0
    if paired:
        simulator.matches = [Match(0, 1, 0, "direct")]
    simulator.solos = [Solo(traveler, 0) for traveler in range(paired, trips)]
    spec = {
        "experiment_id": "custom_run", "mode": "direct", "baseline_id": "custom_run",
        "parameter": "baseline", "baseline_value": None, "value": None,
        "config": config.to_dict(),
    }
    return spec, simulator


def score_tables() -> RunTables:
    return RunTables(
        pd.DataFrame(
            {
                "trip_id": [0, 1, 2], "role": ["driver", "rider", "solo"],
                "delta_min": [2.0, 4.0, 100.0], "wait_min": [0.0, 6.0, 100.0],
                "access_min": [0.0, 2.0, 100.0],
            }
        ),
        pd.DataFrame({"vehicle_min_saved": [7.0]}),
        gpd.GeoDataFrame(geometry=[]),
    )


def test_metrics_measure_matching_counts_and_only_simulation_time(tmp_path) -> None:
    spec, simulator = completed_run()
    metrics = write_experiment_metrics(tmp_path, spec, simulator, 2.0, score_tables())

    assert metrics["trip_count"] == 3
    assert metrics["matched_pairs"] == 1
    assert metrics["matched_travelers"] == 2
    assert metrics["solo_travelers"] == 1
    assert metrics["match_rate"] == pytest.approx(2 / 3)
    assert metrics["simulation_seconds"] == 2.0
    assert metrics["travelers_per_second"] == 1.5
    assert metrics["matched_pairs_per_second"] == 0.5
    assert metrics["seconds_per_matched_pair"] == 2.0
    assert metrics["vehicle_min_saved"] == 7.0
    assert metrics["mean_delta_min"] == 3.0
    assert metrics["mean_wait_min"] == 3.0
    assert metrics["mean_access_min"] == 1.0
    assert metrics["timezone"] == "UTC"
    assert metrics["timestamp"].endswith("+00:00")
    assert metrics["python"]
    assert metrics["platform"]
    assert metrics["config"] == spec["config"]
    expected_hash = hashlib.sha256(
        json.dumps(spec["config"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert metrics["config_sha256"] == expected_hash
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics


def test_zero_matches_have_zero_rate_and_undefined_pair_cost(tmp_path) -> None:
    spec, simulator = completed_run(matched=False)
    tables = RunTables(
        pd.DataFrame({
            "trip_id": [0, 1, 2], "role": ["solo"] * 3,
            "delta_min": [0.0] * 3, "wait_min": [0.0] * 3, "access_min": [0.0] * 3,
        }),
        pd.DataFrame({"vehicle_min_saved": pd.Series(dtype=float)}),
        gpd.GeoDataFrame(geometry=[]),
    )
    metrics = write_experiment_metrics(tmp_path, spec, simulator, 4, tables)

    assert metrics["matched_pairs"] == 0
    assert metrics["match_rate"] == 0.0
    assert metrics["matched_pairs_per_second"] == 0.0
    assert metrics["seconds_per_matched_pair"] is None
    assert metrics["vehicle_min_saved"] == 0.0
    assert metrics["mean_delta_min"] is None
    assert metrics["mean_wait_min"] is None
    assert metrics["mean_access_min"] is None


def test_empty_run_and_zero_timer_remain_valid_json(tmp_path) -> None:
    spec, simulator = completed_run(trips=0)
    metrics = write_experiment_metrics(tmp_path, spec, simulator, 0)

    assert metrics["trip_count"] == 0
    assert metrics["match_rate"] == 0.0
    assert metrics["travelers_per_second"] is None
    assert metrics["matched_pairs_per_second"] is None
    assert metrics["seconds_per_matched_pair"] is None
    assert metrics["vehicle_min_saved"] is None
    assert "NaN" not in (tmp_path / "metrics.json").read_text()


@pytest.mark.parametrize("elapsed", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_elapsed_time_is_rejected_before_writing(tmp_path, elapsed) -> None:
    spec, simulator = completed_run()
    with pytest.raises(ValueError, match="finite nonnegative"):
        write_experiment_metrics(tmp_path, spec, simulator, elapsed)
    assert not (tmp_path / "metrics.json").exists()


def test_incomplete_or_duplicate_coverage_cannot_be_logged(tmp_path) -> None:
    spec, simulator = completed_run()
    simulator.solos = []
    with pytest.raises(ValueError, match="every traveler"):
        write_experiment_metrics(tmp_path, spec, simulator, 1)
    simulator.solos = [Solo(0, 0), Solo(2, 0)]
    with pytest.raises(ValueError, match="more than once"):
        write_experiment_metrics(tmp_path, spec, simulator, 1)
    assert not (tmp_path / "metrics.json").exists()


def test_metric_config_and_score_tables_must_belong_to_this_run(tmp_path) -> None:
    spec, simulator = completed_run()
    spec["config"]["horizon"] = 30
    with pytest.raises(ValueError, match="configuration does not match"):
        write_experiment_metrics(tmp_path, spec, simulator, 1)
    spec["config"]["horizon"] = 15
    tables = score_tables()
    tables.travelers.loc[2, "trip_id"] = 50
    with pytest.raises(ValueError, match="simulation travelers"):
        write_experiment_metrics(tmp_path, spec, simulator, 1, tables)
    assert not (tmp_path / "metrics.json").exists()


@pytest.mark.parametrize("mode", ["direct", "pnr"])
def test_metrics_record_the_actual_router_and_snap_cache_policy(line_network, tmp_path, mode) -> None:
    nodes, edges = line_network
    router = MemoizedRoadRouter(nodes, edges)
    filename = "gt_2022.yaml" if mode == "direct" else "gt_2022_pnr.yaml"
    config = load_config(ROOT / "config" / filename)
    cluster = TripCluster([], [], [], [], [])
    simulator = config.build(cluster, router=router, lots=([0.0], [0.0]))
    spec = {
        "experiment_id": mode, "mode": mode, "baseline_id": mode,
        "parameter": "baseline", "baseline_value": None, "value": None,
        "config": config.to_dict(),
    }
    metrics = write_experiment_metrics(tmp_path, spec, simulator, 1)

    assert metrics["router_class"] == "MemoizedRoadRouter"
    assert metrics["nearest_node_cache_policy"] == "per_instance_exact_coordinates"
    assert router._nearest_node_cache == {}


def test_collection_is_sorted_and_reads_only_immediate_runs(tmp_path) -> None:
    spec, simulator = completed_run()
    spec["experiment_id"] = "e02_direct"
    write_experiment_metrics(tmp_path / "a", spec, simulator, 2)
    spec["experiment_id"] = "e01_direct"
    write_experiment_metrics(tmp_path / "z", spec, simulator, 1)
    write_experiment_metrics(tmp_path / "nested" / "ignored", spec, simulator, 10)
    (tmp_path / "unfinished").mkdir()

    frame = collect_experiment_metrics(tmp_path)

    assert frame["experiment_id"].tolist() == ["e01_direct", "e02_direct"]
    assert frame["simulation_seconds"].tolist() == [1.0, 2.0]
    assert frame["matched_pairs"].tolist() == [1, 1]
    assert frame["output_directory"].tolist() == [str(tmp_path / "z"), str(tmp_path / "a")]
    pd.testing.assert_frame_equal(frame, collect_experiment_metrics(tmp_path))


def test_collection_preserves_corrupt_file_failures_and_empty_columns(tmp_path) -> None:
    empty = collect_experiment_metrics(tmp_path)
    assert empty.empty
    assert {"matched_pairs", "simulation_seconds", "config_sha256"} <= set(empty.columns)
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "metrics.json").write_text("[]")
    with pytest.raises(ValueError, match="invalid experiment metrics"):
        collect_experiment_metrics(tmp_path)
