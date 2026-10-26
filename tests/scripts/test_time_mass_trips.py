"""Sampling and memory-cap rules for the mass-trip speed run."""

from pathlib import Path

import numpy as np
import pandas as pd

from carpoolsim.config import load_config
from scripts.time_mass_trips import PNR_TENSOR_BYTES
from scripts.time_mass_trips import PNR_TENSOR_COPIES
from scripts.time_mass_trips import nested_samples
from scripts.time_mass_trips import output_dir
from scripts.time_mass_trips import peak_window
from scripts.time_mass_trips import pnr_group_cap
from scripts.time_mass_trips import stratified_sample
from scripts.time_mass_trips import with_workers

ROOT = Path(__file__).resolve().parents[2]


def test_stratified_sample_keeps_each_period_share() -> None:
    source = pd.DataFrame({"depart_period": [7] * 50 + [8] * 50, "trip_id": range(100)})
    chosen = stratified_sample(source, 10, np.random.default_rng(0))

    assert len(chosen) == 10
    assert chosen["depart_period"].value_counts().to_dict() == {7: 5, 8: 5}
    assert chosen["trip_id"].is_monotonic_increasing


def test_peak_window_counts_trips_inside_one_horizon() -> None:
    assert peak_window(np.array([0, 10, 15, 16, 40]), 15) == 3
    assert peak_window(np.array([]), 15) == 0


def test_nested_samples_are_subsets() -> None:
    source = pd.DataFrame({"depart_period": [7] * 80 + [8] * 80, "trip_id": range(160)})
    drawn = nested_samples(source, [10, 40, 20], np.random.default_rng(0))

    assert sorted(drawn) == [10, 20, 40]
    assert set(drawn[10]["trip_id"]) < set(drawn[20]["trip_id"]) < set(drawn[40]["trip_id"])
    assert set(drawn[40]["trip_id"]) <= set(source["trip_id"])


def test_mass_configs_use_the_short_horizon() -> None:
    expected = {"horizon": 5.0, "dt": 2.5, "max_size": 50}
    for name in ("mass_sov.yaml", "mass_pnr.yaml"):
        config = load_config(ROOT / "config" / name).to_dict()
        for key, value in expected.items():
            assert config[key] == value
        filters = config["modes"][0]["filters"]
        assert filters[0]["delta1"] == 5.0
        assert filters[-1]["delta2"] == 10.0


def test_one_worker_keeps_the_original_speed_folder() -> None:
    assert output_dir(1) == ROOT / "data_outputs" / "trip_mass_speed" / "h5_dt2p5_m50"
    assert output_dir(4).name == "h5_dt2p5_m50_w4"
    assert output_dir(4).parent == output_dir(1).parent


def test_workers_copy_keeps_the_mass_settings() -> None:
    config = load_config(ROOT / "config" / "mass_sov.yaml")
    parallel = with_workers(config, 4)

    assert config.workers == 1
    assert parallel.workers == 4
    assert parallel.horizon == config.horizon
    assert parallel.dt == config.dt
    assert parallel.max_size == config.max_size
    assert parallel.assigner == config.assigner
    assert [mode.name for mode in parallel.modes] == [mode.name for mode in config.modes]


def test_pnr_group_cap_shrinks_a_window_that_does_not_fit() -> None:
    assert pnr_group_cap(10, n_lots=100, budget=10**12) is None
    cap = pnr_group_cap(10_000, n_lots=3_473, budget=6 * 10**9)
    assert cap is not None
    assert 2 <= cap < 10_000
    assert PNR_TENSOR_BYTES == 4
    assert cap * cap * 3_473 * PNR_TENSOR_BYTES * PNR_TENSOR_COPIES <= 6 * 10**9
