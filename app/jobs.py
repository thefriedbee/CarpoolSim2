"""Load inputs, edit a clock, and run one simulation.

This module does not import Streamlit. The pages call it.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from carpoolsim.config import SimulationConfig
from carpoolsim.map_loader import load_lots
from carpoolsim.map_loader import load_network
from carpoolsim.map_loader import lot_coordinates
from carpoolsim.trip_cluster import TripCluster
from evaluation.experiments import MemoizedRoadRouter
from evaluation.score import RunTables
from evaluation.score import load_results
from evaluation.score import write_results


def with_clock(
    config: SimulationConfig,
    horizon: float,
    dt: float,
    assigner: str,
) -> SimulationConfig:
    """Return a copy with a new horizon, step, and assigner.

    Modes and filters stay as they are in ``config``. ``horizon`` and ``dt``
    are minutes. ``assigner`` is ``matching``, ``greedy``, or ``bipartite``.
    """
    data = config.to_dict()
    data["horizon"] = float(horizon)
    data["dt"] = float(dt)
    data["assigner"] = assigner
    return SimulationConfig.from_dict(data)


def load_inputs(project: Path):
    """Load the Georgia Tech network, trips, and park-and-ride lots.

    The return value is ``(cluster, router, edges, lots)``. ``lots`` is
    ``(px, py)`` in EPSG:3857 meters, ready for ``SimulationConfig.build``.
    """
    network = project / "data_inputs" / "network"
    trips = project / "data_inputs" / "trip" / "gt_2022_off_campus_trips.csv"
    required = (
        network / "nodes.gpkg",
        network / "edges.gpkg",
        network / "lots.gpkg",
        trips,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        names = ", ".join(str(path.relative_to(project)) for path in missing)
        raise FileNotFoundError(f"missing input files: {names}")
    nodes, edges = load_network(network)
    router = MemoizedRoadRouter(nodes, edges, cached=False)
    cluster = TripCluster.from_csv(trips)
    lots = lot_coordinates(load_lots(network))
    return cluster, router, edges, lots


def run_simulation(
    config: SimulationConfig,
    cluster: TripCluster,
    router: MemoizedRoadRouter,
    edges,
    lots,
    directory: Path,
) -> dict[str, object]:
    """Match the day, write the result tables, and return the headline counts."""
    simulator = config.build(cluster, router=router, lots=lots)
    started = perf_counter()
    simulator.run()
    seconds = perf_counter() - started
    tables = write_results(directory, simulator, router, edges)
    summary = summarize(tables.travelers, tables.matches)
    summary["seconds"] = seconds
    summary["directory"] = str(directory)
    return summary


def read_results(directory: str | Path) -> RunTables:
    """Read travelers, carpools, and leg lines from one output folder."""
    return load_results(directory)


def summarize(travelers: pd.DataFrame, matches: pd.DataFrame) -> dict[str, object]:
    """Count travelers, carpools, the match rate, and minutes of driving saved."""
    count = int(len(travelers))
    solo = int((travelers["role"] == "solo").sum()) if count else 0
    matched = count - solo
    saved = 0.0 if matches.empty else float(matches["vehicle_min_saved"].sum())
    return {
        "travelers": count,
        "carpools": int(len(matches)),
        "solo": solo,
        "matched": matched,
        "match_rate": (matched / count) if count else 0.0,
        "minutes_saved": saved,
    }


def summarize_saved(directory: Path) -> dict[str, object]:
    """Summarize a finished run from its CSV tables."""
    travelers = pd.read_csv(directory / "travelers.csv")
    matches = pd.read_csv(directory / "matches.csv")
    return summarize(travelers, matches)


def experiment_table(project: Path) -> pd.DataFrame | None:
    """Return the finished one-factor runs, or ``None`` when the file is absent."""
    path = project / "data_outputs" / "experiments" / "summary.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    columns = [
        "experiment_id",
        "mode",
        "parameter",
        "baseline_value",
        "value",
        "match_rate",
        "vehicle_min_saved",
        "simulation_seconds",
        "description",
    ]
    return frame[[column for column in columns if column in frame.columns]]
