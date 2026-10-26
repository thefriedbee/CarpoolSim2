"""Time mass-trip samples at the short-horizon carpool setting.

``data_inputs/trip_mass/trip_sov_AM.csv`` is the raw zone table. The simulator
reads ``sov_am_1m_trips.csv``, the coordinate file built from it. This script
draws nested departure-stratified samples of that file and times
``simulator.run()`` only.

The default sizes are 5,000, 10,000, 20,000, 50,000, and 100,000 trips. Each
smaller sample is drawn from the next larger one, so the files nest. SOV is
door-to-door carpool. PNR is park-and-ride. Both use ``config/mass_sov.yaml``
and ``config/mass_pnr.yaml``: a 5-minute horizon, a 2.5-minute step, departure
gap 5, groups of 50, and a 10-minute wait.

One worker writes ``data_outputs/trip_mass_speed/h5_dt2p5_m50/``.
``--workers 4`` writes ``h5_dt2p5_m50_w4`` beside it and leaves the
one-worker files in place. The YAML files stay at one worker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "carpoolsim-matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from carpoolsim.config import SimulationConfig
from carpoolsim.config import load_config
from carpoolsim.map_loader import load_lots
from carpoolsim.map_loader import load_network
from carpoolsim.map_loader import lot_coordinates
from carpoolsim.trip_cluster import TripCluster
from evaluation.experiments import MemoizedRoadRouter

SOURCE = ROOT / "data_inputs" / "trip_mass" / "sov_am_1m_trips.csv"
NETWORK = ROOT / "data_inputs" / "network"
CONFIGS = {
    "sov": ROOT / "config" / "mass_sov.yaml",
    "pnr": ROOT / "config" / "mass_pnr.yaml",
}
COUNTS = (5_000, 10_000, 20_000, 50_000, 100_000)
SEED = 20261024
SPEED_DIR = ROOT / "data_outputs" / "trip_mass_speed"
RUN_NAME = "h5_dt2p5_m50"
SAMPLE_DIR = ROOT / "data_inputs" / "trip_mass" / "h5_dt2p5_m50"
# Three float32 copies of the trips × trips × lots tensor, plus headroom
# for the street graph on a 48 GB machine.
PNR_TENSOR_BUDGET = 6 * 10**9
PNR_TENSOR_COPIES = 3
PNR_TENSOR_BYTES = 4


def stratified_sample(source: pd.DataFrame, target: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw ``target`` rows without replacement, keeping each period's share.

    The draw uses the largest-remainder split, so the period counts sum to
    ``target``. Rows come back in source-file order. Same rule as
    ``data_inputs/trip_mass/build_sov_trips.py``.
    """
    periods = source["depart_period"].to_numpy()
    values, counts = np.unique(periods, return_counts=True)
    total = int(counts.sum())
    if target > total:
        raise ValueError(f"asked for {target} trips but the source has {total}")
    if target < 1:
        raise ValueError("asked for no trips")
    base = (target * counts) // total
    leftover = target - int(base.sum())
    remainder = target * counts - base * total
    base[np.argsort(-remainder, kind="stable")[:leftover]] += 1
    if np.any(base > counts):
        raise ValueError("a departure period does not have enough trips")

    picked: list[np.ndarray] = []
    for period, size in zip(values, base, strict=True):
        pool = np.flatnonzero(periods == period)
        picked.append(rng.choice(pool, size=int(size), replace=False))
    index = np.sort(np.concatenate(picked))
    return source.iloc[index].reset_index(drop=True)


def nested_samples(
    source: pd.DataFrame,
    counts: list[int],
    rng: np.random.Generator,
) -> dict[int, pd.DataFrame]:
    """Draw each smaller count from the next larger sample.

    The largest count is drawn from ``source``. Every smaller count is a
    departure-stratified subset of the sample just larger than it.
    """
    ordered = sorted(set(counts))
    if not ordered:
        raise ValueError("asked for no sample sizes")
    drawn: dict[int, pd.DataFrame] = {}
    pool = source
    for count in reversed(ordered):
        pool = stratified_sample(pool, count, rng)
        drawn[count] = pool
    return drawn


def peak_window(minutes: np.ndarray, horizon: float) -> int:
    """Most trips whose departure lies in any ``[t, t + horizon]`` window."""
    order = np.sort(np.asarray(minutes, dtype=np.float64))
    if len(order) == 0:
        return 0
    end = 0
    peak = 0
    for start, left in enumerate(order):
        limit = left + horizon
        while end < len(order) and order[end] <= limit:
            end += 1
        peak = max(peak, end - start)
    return peak


def output_dir(workers: int) -> Path:
    """Return the speed folder for this worker count.

    One worker keeps ``h5_dt2p5_m50``. A larger count uses
    ``h5_dt2p5_m50_wN`` so the one-worker files stay in place.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be an integer of at least 1")
    name = RUN_NAME if workers == 1 else f"{RUN_NAME}_w{workers}"
    return SPEED_DIR / name


def with_workers(config: SimulationConfig, workers: int) -> SimulationConfig:
    """Return a copy that matches groups with ``workers`` processes.

    The YAML files stay at one worker. The other settings are unchanged.
    """
    return SimulationConfig(
        horizon=config.horizon,
        dt=config.dt,
        et=config.et,
        commit=config.commit,
        assigner=config.assigner,
        modes=config.modes,
        max_size=config.max_size,
        workers=workers,
    )


def pnr_group_cap(peak: int, n_lots: int, budget: int = PNR_TENSOR_BUDGET) -> int | None:
    """Group size that keeps the lot tensors inside ``budget``, or None.

    ``None`` means the whole window already fits. The cap is at least 2.
    """
    if peak < 2 or n_lots < 1:
        return None
    need = peak * peak * n_lots * PNR_TENSOR_BYTES * PNR_TENSOR_COPIES
    if need <= budget:
        return None
    room = budget / (PNR_TENSOR_COPIES * n_lots * PNR_TENSOR_BYTES)
    cap = int(np.floor(np.sqrt(room)))
    return max(2, min(cap, peak))


def time_run(
    config: SimulationConfig,
    trips: Path,
    *,
    lots: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, object]:
    """Load a fresh router and return the simulation-only timing."""
    started = perf_counter()
    nodes, edges = load_network(NETWORK)
    router = MemoizedRoadRouter(nodes, edges, cached=False)
    cluster = TripCluster.from_csv(trips)
    load_seconds = perf_counter() - started
    started = perf_counter()
    simulator = config.build(cluster, router=router, lots=lots)
    build_seconds = perf_counter() - started
    started = perf_counter()
    simulator.run()
    simulation_seconds = perf_counter() - started
    delta1, delta2 = _wait_limits(config)
    return {
        "trip_count": len(cluster),
        "matched_pairs": len(simulator.matches),
        "solo_travelers": len(simulator.solos),
        "load_seconds": load_seconds,
        "build_seconds": build_seconds,
        "simulation_seconds": simulation_seconds,
        "horizon": config.horizon,
        "dt": config.dt,
        "delta1": delta1,
        "delta2": delta2,
        "assigner": config.assigner,
        "max_size": config.max_size,
        "workers": config.workers,
        "modes": [mode.name for mode in config.modes],
    }


def _wait_limits(config: SimulationConfig) -> tuple[float | None, float | None]:
    """Return the departure gap and the wait limit recorded on the first mode."""
    delta1 = None
    delta2 = None
    for spec in config.to_dict()["modes"][0]["filters"]:
        if "delta1" in spec:
            delta1 = spec["delta1"]
        if "delta2" in spec:
            delta2 = spec["delta2"]
    return delta1, delta2


def _write_summary(out_dir: Path) -> None:
    """Rewrite ``summary.csv`` from the JSON files already saved."""
    rows = []
    for path in sorted(out_dir.glob("n*/sov.json")) + sorted(out_dir.glob("n*/pnr.json")):
        rows.append(json.loads(path.read_text()))
    if not rows:
        return
    frame = pd.DataFrame(rows)
    columns = [
        "trip_count", "mode", "workers", "simulation_seconds", "matched_pairs", "solo_travelers",
        "peak_window", "horizon", "dt", "delta1", "delta2", "max_size", "load_seconds",
    ]
    frame = frame.sort_values(["trip_count", "mode"])
    frame.to_csv(out_dir / "summary.csv", index=False, columns=[name for name in columns if name in frame.columns])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, nargs="+", default=list(COUNTS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--modes", nargs="+", choices=("sov", "pnr"), default=("sov", "pnr"))
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Processes that match groups. One writes h5_dt2p5_m50; more write h5_dt2p5_m50_wN.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be an integer of at least 1")

    out_dir = output_dir(args.workers)
    configs = {mode: with_workers(load_config(CONFIGS[mode]), args.workers) for mode in args.modes}
    out_dir.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"settings: horizon=5, dt=2.5, delta1=5, max_size=50, delta2=10, workers={args.workers}",
        flush=True,
    )
    print(f"writing {out_dir.relative_to(ROOT)}", flush=True)
    print(f"reading {SOURCE.name}", flush=True)
    source = pd.read_csv(SOURCE)
    samples = nested_samples(source, args.count, np.random.default_rng(args.seed))
    lots = None
    if "pnr" in args.modes:
        lot_table = load_lots(NETWORK)
        lots = lot_coordinates(lot_table)
        print(f"parking lots: {len(lot_table):,}", flush=True)

    for count in sorted(samples):
        sample = samples[count]
        sample_path = SAMPLE_DIR / f"sov_am_{count}_trips.csv"
        sample.to_csv(sample_path, index=False)
        minutes = np.array([_minutes(clock) for clock in sample["departure_time"]])
        horizon = next(iter(configs.values())).horizon
        peak = peak_window(minutes, horizon)
        run_dir = out_dir / f"n{count}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"wrote {sample_path.relative_to(ROOT)}: {len(sample):,} trips, "
            f"peak {horizon:.0f}-minute window {peak:,}",
            flush=True,
        )
        for mode in args.modes:
            config = configs[mode]
            print(
                f"starting n={count:,} {mode}: horizon={config.horizon}, dt={config.dt}, "
                f"max_size={config.max_size}, workers={config.workers}",
                flush=True,
            )
            result = time_run(config, sample_path, lots=lots if mode == "pnr" else None)
            result.update(
                {
                    "mode": mode,
                    "sample": str(sample_path.relative_to(ROOT)),
                    "source": str(SOURCE.relative_to(ROOT)),
                    "seed": args.seed,
                    "peak_window": peak,
                }
            )
            path = run_dir / f"{mode}.json"
            path.write_text(json.dumps(result, indent=2) + "\n")
            _write_summary(out_dir)
            print(
                f"n={count:,} {mode}: {result['simulation_seconds']:.1f}s simulation, "
                f"{result['matched_pairs']} pairs, {result['solo_travelers']} solo. "
                f"wrote {path.relative_to(ROOT)}",
                flush=True,
            )


def _minutes(clock: str) -> int:
    hour, minute = clock.split(":")
    return int(hour) * 60 + int(minute)


if __name__ == "__main__":
    main()
