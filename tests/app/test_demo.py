"""Tests for the demo helpers and the one-way import boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.jobs import experiment_table
from app.jobs import summarize_saved
from app.paths import example_run
from app.paths import saved_runs

ROOT = Path(__file__).resolve().parents[2]


def test_example_run_is_the_georgia_tech_folder() -> None:
    folder = example_run()

    assert folder.parts[-2:] == ("data_outputs", "gt_2022")


def test_saved_runs_include_the_door_to_door_results() -> None:
    folder = example_run()
    if not (folder / "travelers.csv").is_file():
        pytest.skip("local Georgia Tech results are not in this checkout")

    found = {path.resolve() for path in saved_runs()}

    assert folder.resolve() in found


def test_saved_summary_counts_the_door_to_door_run() -> None:
    folder = example_run()
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

    assert len(table) == 75
    assert "e01_direct_baseline" in set(table["experiment_id"])
    assert "e21_both_direct_first" in set(table["experiment_id"])
    assert "e75_pnr_mu_1_2" in set(table["experiment_id"])


def test_core_packages_do_not_import_the_app() -> None:
    for package in ("carpoolsim", "evaluation"):
        for path in (ROOT / package).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "streamlit" not in text
            assert "import app" not in text
            assert "from app" not in text


def test_result_helpers_do_not_import_streamlit() -> None:
    for name in ("jobs.py", "paths.py"):
        text = (ROOT / "app" / name).read_text(encoding="utf-8")
        assert "streamlit" not in text
