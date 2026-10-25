"""One-factor experiment plans and simulation-only speed measurements."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
from datetime import datetime
from datetime import timezone
from numbers import Real
from pathlib import Path

import geopandas as gpd
import pandas as pd

from carpoolsim.config import SimulationConfig
from carpoolsim.config import load_config
from carpoolsim.network_prep.partition import NetworkPartition
from carpoolsim.network_prep.router import RoadRouter
from carpoolsim.simulator import Simulator
from evaluation.score import RunTables


class MemoizedRoadRouter(RoadRouter):
    """Memoize exact point-to-node lookups within one experiment.

    The cache starts empty and the first lookup delegates to
    ``RoadRouter.nearest_node``. Repeated coordinates reuse that exact node
    result. Use a fresh instance for every measured simulation so its first
    lookups are included in the timer. The portal path cache is on unless
    ``cached`` is false.
    """

    nearest_node_cache_policy = "per_instance_exact_coordinates"

    def __init__(
        self,
        nodes: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        *,
        cached: bool = True,
        partition: NetworkPartition | None = None,
        max_nodes: int = 400,
        min_share: float = 0.30,
    ) -> None:
        super().__init__(
            nodes, edges, cached=cached, partition=partition,
            max_nodes=max_nodes, min_share=min_share,
        )
        self._nearest_node_cache: dict[tuple[float, float], int] = {}

    def nearest_node(self, lon: float, lat: float) -> int:
        """Return the base router's nearest node, computing each point once."""
        key = (lon, lat)
        if key not in self._nearest_node_cache:
            self._nearest_node_cache[key] = super().nearest_node(lon, lat)
        return self._nearest_node_cache[key]


def _changed_config(
    baseline: dict,
    parameter: str,
    value: object,
    label: str,
) -> tuple[dict, object]:
    """Return a validated config and the previous leaf after one change."""
    config = copy.deepcopy(baseline)
    target = config
    parts = parameter.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    key = int(parts[-1]) if isinstance(target, list) else parts[-1]
    previous = target[key]
    if previous == value:
        raise ValueError(f"experiment {label} does not change {parameter}")
    target[key] = value
    return SimulationConfig.from_dict(config).to_dict(), previous


def experiment_specs(root: str | Path) -> list[dict[str, object]]:
    """Return the one-factor screen, the joint baseline, then level sweeps.

    The first 20 runs are two baselines and nine changes per mode. Each of
    those starts from its mode's current ``config/gt_2022*.yaml`` file. The
    next run places direct carpool before park-and-ride and keeps each mode's
    baseline filters. The two mode files must already share one clock and
    assigner, because a joint run has one of each. Those 21 ids stay fixed.

    Later runs fill five-level curves for the controls that moved matching
    size in the screen. Each new run still changes one leaf from its mode
    baseline. The Euclidean detour ratio stays at the original two points.
    The baseline value and the screen value are not repeated. Runs ``e61``
    through ``e69`` then give the horizon five levels and give both modes
    matching, bipartite, and greedy assignment. Ids ``e01`` through ``e60``
    stay fixed.

    Times are minutes; ``r1`` and ``r_p`` are kilometers; the other filter
    controls are ratios. A larger ``dt`` checks the pool less frequently.
    ``parameter`` is a dot path into ``config`` (list indices are integers).
    The baseline records ``parameter='baseline'`` and null values.
    """
    direct = [
        ("horizon_30", "horizon", 30, "Expand the matching horizon from 15 to 30 minutes."),
        ("dt_10", "dt", 10, "Check the pool every 10 minutes instead of every 5 minutes."),
        ("delta1_5", "modes.0.filters.0.delta1", 5, "Limit departure separation to 5 minutes."),
        ("r1_2p5", "modes.0.filters.1.r1", 2.5, "Limit origin separation to 2.5 kilometers."),
        ("mu_1_1p25", "modes.0.filters.2.mu_1", 1.25, "Limit the Euclidean detour ratio to 1.25."),
        ("gamma_1_1p25", "modes.0.filters.3.gamma_1", 1.25, "Limit the driving-time ratio to 1.25."),
        ("gamma_2_0p75", "modes.0.filters.4.gamma_2", 0.75, "Require at least a 0.75 shared-time ratio."),
        ("delta2_5", "modes.0.filters.5.delta2", 5, "Limit passenger waiting to 5 minutes."),
        ("assigner_greedy", "assigner", "greedy", "Compare greedy assignment with maximum-cardinality matching."),
    ]
    pnr = [
        ("horizon_30", "horizon", 30, "Expand the matching horizon from 15 to 30 minutes."),
        ("dt_10", "dt", 10, "Check the pool every 10 minutes instead of every 5 minutes."),
        ("delta1_5", "modes.0.filters.0.delta1", 5, "Limit departure separation to 5 minutes."),
        ("r_p_2p5", "modes.0.r_p", 2.5, "Limit origin-to-lot reach to 2.5 kilometers."),
        ("mu_1_1p25", "modes.0.filters.2.mu_1", 1.25, "Limit the Euclidean detour ratio to 1.25."),
        ("mu_3_0p25", "modes.0.filters.3.mu_3", 0.25, "Limit the passenger access-distance ratio to 0.25."),
        ("gamma_1_1p25", "modes.0.filters.4.gamma_1", 1.25, "Limit the driving-time ratio to 1.25."),
        ("gamma_2_0p75", "modes.0.filters.5.gamma_2", 0.75, "Require at least a 0.75 shared-time ratio."),
        ("delta2_5", "modes.0.filters.6.delta2", 5, "Limit waiting at the lot to 5 minutes."),
    ]
    direct_sweep = [
        ("dt_1", "dt", 1, "Check the pool every 1 minute."),
        ("dt_3", "dt", 3, "Check the pool every 3 minutes."),
        ("dt_15", "dt", 15, "Check the pool every 15 minutes."),
        ("delta1_10", "modes.0.filters.0.delta1", 10, "Set departure separation to 10 minutes."),
        ("delta1_20", "modes.0.filters.0.delta1", 20, "Set departure separation to 20 minutes."),
        ("delta1_30", "modes.0.filters.0.delta1", 30, "Set departure separation to 30 minutes."),
        ("r1_1", "modes.0.filters.1.r1", 1, "Set origin separation to 1 kilometer."),
        ("r1_10", "modes.0.filters.1.r1", 10, "Set origin separation to 10 kilometers."),
        ("r1_15", "modes.0.filters.1.r1", 15, "Set origin separation to 15 kilometers."),
        ("gamma_1_1p1", "modes.0.filters.3.gamma_1", 1.1, "Set the driving-time ratio to 1.1."),
        ("gamma_1_2", "modes.0.filters.3.gamma_1", 2, "Set the driving-time ratio to 2."),
        ("gamma_1_2p5", "modes.0.filters.3.gamma_1", 2.5, "Set the driving-time ratio to 2.5."),
        ("gamma_2_0p25", "modes.0.filters.4.gamma_2", 0.25, "Set the shared-time ratio to 0.25."),
        ("gamma_2_0p9", "modes.0.filters.4.gamma_2", 0.9, "Set the shared-time ratio to 0.9."),
        ("gamma_2_1", "modes.0.filters.4.gamma_2", 1, "Set the shared-time ratio to 1."),
        ("delta2_0", "modes.0.filters.5.delta2", 0, "Set passenger waiting to 0 minutes."),
        ("delta2_15", "modes.0.filters.5.delta2", 15, "Set passenger waiting to 15 minutes."),
        ("delta2_20", "modes.0.filters.5.delta2", 20, "Set passenger waiting to 20 minutes."),
    ]
    pnr_sweep = [
        ("dt_1", "dt", 1, "Check the pool every 1 minute."),
        ("dt_3", "dt", 3, "Check the pool every 3 minutes."),
        ("dt_15", "dt", 15, "Check the pool every 15 minutes."),
        ("delta1_10", "modes.0.filters.0.delta1", 10, "Set departure separation to 10 minutes."),
        ("delta1_20", "modes.0.filters.0.delta1", 20, "Set departure separation to 20 minutes."),
        ("delta1_30", "modes.0.filters.0.delta1", 30, "Set departure separation to 30 minutes."),
        ("r_p_1", "modes.0.r_p", 1, "Set origin-to-lot reach to 1 kilometer."),
        ("r_p_10", "modes.0.r_p", 10, "Set origin-to-lot reach to 10 kilometers."),
        ("r_p_15", "modes.0.r_p", 15, "Set origin-to-lot reach to 15 kilometers."),
        ("mu_3_0p1", "modes.0.filters.3.mu_3", 0.1, "Set the passenger access-distance ratio to 0.1."),
        ("mu_3_0p75", "modes.0.filters.3.mu_3", 0.75, "Set the passenger access-distance ratio to 0.75."),
        ("mu_3_1", "modes.0.filters.3.mu_3", 1, "Set the passenger access-distance ratio to 1."),
        ("gamma_1_1p1", "modes.0.filters.4.gamma_1", 1.1, "Set the driving-time ratio to 1.1."),
        ("gamma_1_2", "modes.0.filters.4.gamma_1", 2, "Set the driving-time ratio to 2."),
        ("gamma_1_2p5", "modes.0.filters.4.gamma_1", 2.5, "Set the driving-time ratio to 2.5."),
        ("gamma_2_0p25", "modes.0.filters.5.gamma_2", 0.25, "Set the shared-time ratio to 0.25."),
        ("gamma_2_0p9", "modes.0.filters.5.gamma_2", 0.9, "Set the shared-time ratio to 0.9."),
        ("gamma_2_1", "modes.0.filters.5.gamma_2", 1, "Set the shared-time ratio to 1."),
        ("delta2_0", "modes.0.filters.6.delta2", 0, "Set waiting at the lot to 0 minutes."),
        ("delta2_15", "modes.0.filters.6.delta2", 15, "Set waiting at the lot to 15 minutes."),
        ("delta2_20", "modes.0.filters.6.delta2", 20, "Set waiting at the lot to 20 minutes."),
    ]
    # Appended after e60 so the earlier ids stay attached to their outputs.
    added = [
        ("direct", "horizon_5", "horizon", 5, "Set the matching horizon to 5 minutes."),
        ("direct", "horizon_10", "horizon", 10, "Set the matching horizon to 10 minutes."),
        ("direct", "horizon_45", "horizon", 45, "Set the matching horizon to 45 minutes."),
        ("pnr", "horizon_5", "horizon", 5, "Set the matching horizon to 5 minutes."),
        ("pnr", "horizon_10", "horizon", 10, "Set the matching horizon to 10 minutes."),
        ("pnr", "horizon_45", "horizon", 45, "Set the matching horizon to 45 minutes."),
        (
            "direct",
            "assigner_bipartite",
            "assigner",
            "bipartite",
            "Compare bipartite assignment with maximum-cardinality matching.",
        ),
        (
            "pnr",
            "assigner_greedy",
            "assigner",
            "greedy",
            "Compare greedy assignment with maximum-cardinality matching.",
        ),
        (
            "pnr",
            "assigner_bipartite",
            "assigner",
            "bipartite",
            "Compare bipartite assignment with maximum-cardinality matching.",
        ),
    ]
    specs: list[dict[str, object]] = []
    baselines: dict[str, tuple[str, dict]] = {}
    for mode, filename, variants in (
        ("direct", "gt_2022.yaml", direct),
        ("pnr", "gt_2022_pnr.yaml", pnr),
    ):
        baseline = load_config(Path(root) / "config" / filename).to_dict()
        baseline_id = f"e{len(specs) + 1:02d}_{mode}_baseline"
        specs.append(
            {
                "experiment_id": baseline_id,
                "mode": mode,
                "baseline_id": baseline_id,
                "parameter": "baseline",
                "baseline_value": None,
                "value": None,
                "description": f"Unchanged {mode} configuration from config/{filename}.",
                "config": copy.deepcopy(baseline),
            }
        )
        baselines[mode] = (baseline_id, baseline)
        for suffix, parameter, value, description in variants:
            config, previous = _changed_config(baseline, parameter, value, f"{mode}/{suffix}")
            specs.append(
                {
                    "experiment_id": f"e{len(specs) + 1:02d}_{mode}_{suffix}",
                    "mode": mode,
                    "baseline_id": baseline_id,
                    "parameter": parameter,
                    "baseline_value": previous,
                    "value": value,
                    "description": description,
                    "config": config,
                }
            )
    specs.append(_both_direct_first(root))
    for mode, variants in (("direct", direct_sweep), ("pnr", pnr_sweep)):
        baseline_id, baseline = baselines[mode]
        for suffix, parameter, value, description in variants:
            config, previous = _changed_config(baseline, parameter, value, f"{mode}/{suffix}")
            specs.append(
                {
                    "experiment_id": f"e{len(specs) + 1:02d}_{mode}_{suffix}",
                    "mode": mode,
                    "baseline_id": baseline_id,
                    "parameter": parameter,
                    "baseline_value": previous,
                    "value": value,
                    "description": description,
                    "config": config,
                }
            )
    for mode, suffix, parameter, value, description in added:
        baseline_id, baseline = baselines[mode]
        config, previous = _changed_config(baseline, parameter, value, f"{mode}/{suffix}")
        specs.append(
            {
                "experiment_id": f"e{len(specs) + 1:02d}_{mode}_{suffix}",
                "mode": mode,
                "baseline_id": baseline_id,
                "parameter": parameter,
                "baseline_value": previous,
                "value": value,
                "description": description,
                "config": config,
            }
        )
    return specs


def _both_direct_first(root: str | Path) -> dict[str, object]:
    """Direct carpool first, then park-and-ride, both at their baseline filters."""
    direct = load_config(Path(root) / "config" / "gt_2022.yaml").to_dict()
    pnr = load_config(Path(root) / "config" / "gt_2022_pnr.yaml").to_dict()
    shared = ("horizon", "dt", "et", "commit", "assigner", "max_size")
    for key in shared:
        if direct.get(key) != pnr.get(key):
            raise ValueError(f"joint baseline requires the same {key} in both mode files")
    config = {key: copy.deepcopy(direct[key]) for key in shared if key in direct}
    config["modes"] = [copy.deepcopy(direct["modes"][0]), copy.deepcopy(pnr["modes"][0])]
    config = SimulationConfig.from_dict(config).to_dict()
    return {
        "experiment_id": "e21_both_direct_first",
        "mode": "both",
        "baseline_id": "e21_both_direct_first",
        "parameter": "baseline",
        "baseline_value": None,
        "value": None,
        "description": (
            "Direct carpool, then park-and-ride, each with its unchanged baseline filters."
        ),
        "config": config,
    }


def write_experiment_metrics(
    directory: str | Path,
    spec: dict[str, object],
    simulator: Simulator,
    simulation_seconds: float,
    tables: RunTables | None = None,
) -> dict[str, object]:
    """Save one completed run's counts, speed, configuration, and scores.

    Pass elapsed wall time measured only around ``simulator.run()``. Loading,
    scoring, plotting, and writing results must happen outside that timer.
    ``matched_pairs`` counts cars; ``matched_travelers`` counts both seats.
    Match rate is the fraction of all travelers who matched. Mean delay,
    waiting, and access times include matched travelers only. Missing scores
    and undefined divisions are JSON null. An elapsed time of zero is allowed
    but cannot define a throughput. Incomplete or repeated traveler coverage
    and a configuration different from ``spec['config']`` are rejected.
    """
    if (
        isinstance(simulation_seconds, bool)
        or not isinstance(simulation_seconds, Real)
        or not math.isfinite(simulation_seconds)
        or simulation_seconds < 0
    ):
        raise ValueError("simulation_seconds must be a finite nonnegative number")
    elapsed = float(simulation_seconds)
    config = SimulationConfig.from_simulator(simulator).to_dict()
    expected = SimulationConfig.from_dict(spec["config"]).to_dict()
    if config != expected:
        raise ValueError("simulator configuration does not match experiment config")

    recorded = [int(trip_id) for match in simulator.matches for trip_id in (match.driver, match.rider)]
    recorded.extend(int(solo.traveler) for solo in simulator.solos)
    if len(recorded) != len(set(recorded)):
        raise ValueError("a traveler is matched or solo more than once")
    if set(recorded) != {int(trip_id) for trip_id in simulator.cluster.ids}:
        raise ValueError("every traveler must be matched or solo before logging metrics")

    trip_count = len(simulator.cluster)
    matched_pairs = len(simulator.matches)
    matched_travelers = 2 * matched_pairs
    router = _simulation_router(simulator)
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    metrics = {
        key: spec[key]
        for key in ("experiment_id", "mode", "baseline_id", "parameter", "baseline_value", "value")
    }
    metrics.update(
        {
            "status": "complete",
            "trip_count": trip_count,
            "matched_pairs": matched_pairs,
            "matched_travelers": matched_travelers,
            "solo_travelers": len(simulator.solos),
            "match_rate": matched_travelers / trip_count if trip_count else 0.0,
            "simulation_seconds": elapsed,
            "travelers_per_second": trip_count / elapsed if elapsed else None,
            "matched_pairs_per_second": matched_pairs / elapsed if elapsed else None,
            "seconds_per_matched_pair": elapsed / matched_pairs if matched_pairs else None,
            "vehicle_min_saved": None,
            "mean_delta_min": None,
            "mean_wait_min": None,
            "mean_access_min": None,
            "config_sha256": digest,
            "config": config,
            "horizon": config["horizon"],
            "dt": config["dt"],
            "et": config["et"],
            "commit": config["commit"],
            "assigner": config["assigner"],
            "router_class": type(router).__name__ if router is not None else None,
            "nearest_node_cache_policy": getattr(router, "nearest_node_cache_policy", "none"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timezone": "UTC",
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
    )
    if tables is not None:
        if len(tables.travelers) != trip_count or len(tables.matches) != matched_pairs:
            raise ValueError("score tables do not match the simulation counts")
        if set(tables.travelers["trip_id"].tolist()) != set(recorded):
            raise ValueError("score tables do not match the simulation travelers")
        paired = tables.travelers.loc[tables.travelers["role"].isin(["driver", "rider"])]
        if len(paired) != matched_travelers:
            raise ValueError("score tables do not match the simulation matched travelers")
        metrics["vehicle_min_saved"] = _finite_or_none(tables.matches["vehicle_min_saved"].sum())
        for column in ("delta_min", "wait_min", "access_min"):
            metrics[f"mean_{column}"] = _finite_or_none(paired[column].mean())

    # Strict JSON disallows NaN or infinity, so saved metrics remain portable.
    payload = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(payload, encoding="utf-8")
    return metrics


def collect_experiment_metrics(output_root: str | Path) -> pd.DataFrame:
    """Read immediate child ``metrics.json`` files in experiment-id order.

    Missing runs have no row. Unreadable or malformed files raise rather
    than silently dropping an experiment from the comparison.
    """
    records = []
    for path in sorted(Path(output_root).glob("*/metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or "experiment_id" not in record:
            raise ValueError(f"invalid experiment metrics: {path}")
        records.append({**record, "output_directory": str(path.parent.resolve())})
    if not records:
        return pd.DataFrame(
            columns=[
                "experiment_id", "mode", "baseline_id", "parameter", "baseline_value", "value",
                "status", "trip_count", "matched_pairs", "matched_travelers", "solo_travelers",
                "match_rate", "simulation_seconds", "travelers_per_second",
                "matched_pairs_per_second", "seconds_per_matched_pair", "vehicle_min_saved",
                "mean_delta_min", "mean_wait_min", "mean_access_min", "config_sha256",
                "config", "horizon", "dt", "et", "commit", "assigner", "timestamp", "timezone",
                "router_class", "nearest_node_cache_policy", "python", "platform", "output_directory",
            ]
        )
    return pd.DataFrame(records).sort_values("experiment_id", kind="stable").reset_index(drop=True)


def _finite_or_none(value: Real) -> float | None:
    """Convert aggregate scalars, including NumPy NaN, to strict JSON values."""
    number = float(value)
    return number if math.isfinite(number) else None


def _simulation_router(simulator: Simulator) -> RoadRouter | None:
    """Find the router held by the first filter that uses network travel."""
    for _name, filters in simulator.modes:
        for step in filters:
            router = getattr(step, "router", None)
            if router is None:
                router = getattr(getattr(step, "_travel", None), "router", None)
            if router is not None:
                return router
    return None
