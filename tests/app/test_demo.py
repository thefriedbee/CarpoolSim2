"""Tests for the demo helpers and the one-way import boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.jobs import experiment_table
from app.jobs import summarize_saved
from app.jobs import with_clock
from app.paths import demo_output
from app.paths import presets
from app.paths import saved_runs
from carpoolsim.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def test_with_clock_keeps_the_filters() -> None:
    config = load_config(ROOT / "config" / "gt_2022.yaml")

    edited = with_clock(config, 30, 10, "greedy")

    assert edited.horizon == pytest.approx(30)
    assert edited.dt == pytest.approx(10)
    assert edited.assigner == "greedy"
    assert edited.commit == "window"
    assert edited.modes[0].name == "direct"
    assert edited.modes[0].filters[0].name == "FilterDepartureTime"
    assert edited.modes[0].filters[0].params["delta1"] == pytest.approx(15)


def test_a_step_longer_than_the_horizon_is_rejected() -> None:
    config = load_config(ROOT / "config" / "gt_2022.yaml")

    with pytest.raises(ValueError, match="horizon"):
        with_clock(config, 5, 10, "matching")


def test_presets_start_with_the_two_georgia_tech_files() -> None:
    found = presets()

    assert found[0].name == "gt_2022.yaml"
    assert found[1].name == "gt_2022_pnr.yaml"
    assert any(path.name == "e01_direct_baseline.yaml" for path in found)


def test_demo_output_stays_out_of_the_notebook_folders() -> None:
    folder = demo_output("gt_2022")

    assert folder.parts[-3:] == ("data_outputs", "demo", "gt_2022")


def test_saved_runs_include_the_door_to_door_results() -> None:
    folder = ROOT / "data_outputs" / "gt_2022"
    if not (folder / "travelers.csv").is_file():
        pytest.skip("local Georgia Tech results are not in this checkout")

    found = {path.resolve() for path in saved_runs()}

    assert folder.resolve() in found


def test_saved_summary_counts_the_door_to_door_run() -> None:
    folder = ROOT / "data_outputs" / "gt_2022"
    if not (folder / "travelers.csv").is_file():
        pytest.skip("local Georgia Tech results are not in this checkout")

    summary = summarize_saved(folder)

    assert summary["travelers"] == 1539
    assert summary["carpools"] == 532
    assert summary["solo"] == 475
    assert summary["match_rate"] == pytest.approx(1064 / 1539)


def test_experiment_table_has_one_row_per_finished_run() -> None:
    table = experiment_table(ROOT)
    if table is None:
        pytest.skip("local experiment summary is not in this checkout")

    assert len(table) == 21
    assert "e01_direct_baseline" in set(table["experiment_id"])
    assert "e21_both_direct_first" in set(table["experiment_id"])


def test_core_packages_do_not_import_the_app() -> None:
    for package in ("carpoolsim", "evaluation"):
        for path in (ROOT / package).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "streamlit" not in text
            assert "import app" not in text
            assert "from app" not in text


def test_run_helpers_do_not_import_streamlit() -> None:
    for name in ("jobs.py", "paths.py"):
        text = (ROOT / "app" / name).read_text(encoding="utf-8")
        assert "streamlit" not in text
