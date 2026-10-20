"""Project paths for the demo. This module does not import Streamlit."""

from __future__ import annotations

from pathlib import Path


def root() -> Path:
    """Return the repository root that contains ``pyproject.toml``."""
    return Path(__file__).resolve().parents[1]


def presets() -> list[Path]:
    """Return the door-to-door file, the park-and-ride file, then experiments."""
    project = root()
    ordered = [
        project / "config" / "gt_2022.yaml",
        project / "config" / "gt_2022_pnr.yaml",
    ]
    ordered.extend(sorted((project / "config" / "experiments").glob("*.yaml")))
    return [path for path in ordered if path.is_file()]


def preset_note(path: Path) -> str:
    """Return the leading comments of a config file, as one sentence."""
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            lines.append(line.lstrip("#").strip())
        elif line.strip():
            break
    return " ".join(lines)


def saved_runs() -> list[Path]:
    """Return output folders that contain the three result tables."""
    outputs = root() / "data_outputs"
    if not outputs.is_dir():
        return []
    runs = []
    for travelers in sorted(outputs.rglob("travelers.csv")):
        folder = travelers.parent
        if (folder / "matches.csv").is_file() and (folder / "legs.gpkg").is_file():
            runs.append(folder)
    return runs


def demo_output(stem: str) -> Path:
    """Return the folder where a demo run of ``stem`` is written.

    A second run of the same preset replaces this folder. The original
    notebooks' outputs stay under their own directories.
    """
    return root() / "data_outputs" / "demo" / stem
