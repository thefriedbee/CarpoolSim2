"""Generate the frozen one-factor plan and one notebook per experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "carpoolsim-matplotlib"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.experiments import experiment_specs


SETUP = '''%matplotlib inline

import json
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from IPython.display import display

here = Path.cwd().resolve()
for candidate in [here, *here.parents]:
    if (candidate / "evaluation").is_dir() and (candidate / "pyproject.toml").exists():
        root = candidate
        break
else:
    raise RuntimeError("Run this notebook from the CarpoolSim project or a folder inside it.")
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from carpoolsim.config import load_config
from carpoolsim.map_loader import load_lots, load_network, lot_coordinates
from carpoolsim.trip_cluster import TripCluster
from evaluation import plot_before_after, plot_delta, plot_trip, write_results
from evaluation.experiments import MemoizedRoadRouter, write_experiment_metrics
from scripts.run_experiments import begin_attempt, complete_attempt
'''


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str, **metadata):
    return nbformat.v4.new_code_cell(source.strip(), metadata=metadata)


def notebook(cells):
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
    )


def experiment_notebook(spec):
    experiment_id = spec["experiment_id"]
    baseline = spec["parameter"] == "baseline"
    change = (
        "This is the unchanged baseline for this mode."
        if baseline
        else f"Only `{spec['parameter']}` changes: **{spec['baseline_value']} → {spec['value']}**."
    )
    joint = spec["mode"] == "both"
    number = int(str(experiment_id)[1:3])
    if joint:
        overview = "This run is listed with the other experiments in [summary.ipynb](summary.ipynb)."
    elif number <= 20:
        overview = "View all 20 runs in [summary.ipynb](summary.ipynb)."
    else:
        overview = "View every run in [summary.ipynb](summary.ipynb)."
    prepare_note = (
        "This cell loads every trip, the saved parking lots, and a fresh road router. Direct carpool runs first and removes the travelers it matches. Park-and-ride then matches only the travelers direct left."
        if joint
        else "This cell loads every trip and constructs a fresh road router. PNR runs also load the same saved parking lots."
    )
    prepare_code = '''nodes, edges = load_network(NETWORK_DIR)
router = MemoizedRoadRouter(nodes, edges, cached=True)
cluster = TripCluster.from_csv(TRIP_PATH)
build_options = {"router": router}
lots = load_lots(NETWORK_DIR)
build_options["lots"] = lot_coordinates(lots)
print(f"Parking lots: {len(lots):,}")
simulator = config.build(cluster, **build_options)
print(f"Loaded {len(cluster):,} travelers, {len(nodes):,} nodes, and {len(edges):,} edges.")
''' if joint else '''nodes, edges = load_network(NETWORK_DIR)
router = MemoizedRoadRouter(nodes, edges, cached=True)
cluster = TripCluster.from_csv(TRIP_PATH)
build_options = {"router": router}
if spec["mode"] == "pnr":
    lots = load_lots(NETWORK_DIR)
    build_options["lots"] = lot_coordinates(lots)
    print(f"Parking lots: {len(lots):,}")
simulator = config.build(cluster, **build_options)
print(f"Loaded {len(cluster):,} travelers, {len(nodes):,} nodes, and {len(edges):,} edges.")
'''
    cells = [
            markdown(f'''# {experiment_id}

{spec["description"]}

{change} The comparison baseline is `{spec["baseline_id"]}`. All Georgia Tech 2022 off-campus travelers, road inputs, other settings, and the initial empty exact nearest-node lookup cache are kept consistent across this set of experiments. The portal path cache is on.

The full parameter set is saved in `config/experiments/{experiment_id}.yaml` and displayed below. Results go to `data_outputs/experiments/{experiment_id}`. {overview}

Only the cell containing `simulator.run()` is timed. Loading data, constructing the simulator, writing outputs, scoring, and drawing charts are excluded. Matching size counts carpool pairs; each pair contains two travelers. These are single runs, so runtime comparisons include ordinary machine and cache variation.
'''),
            code(SETUP),
            markdown("## Parameter set\n\nThe saved manifest freezes this experiment and its mode's baseline. The YAML must agree with that frozen parameter set."),
            code(f'''EXPERIMENT_ID = {experiment_id!r}
manifest = json.loads((root / "config/experiments/manifest.json").read_text())
spec = next(item for item in manifest["experiments"] if item["experiment_id"] == EXPERIMENT_ID)
CONFIG = root / spec["config_path"]
OUT_DIR = root / spec["output_path"]
NOTEBOOK = root / spec["notebook_path"]
execution_provenance = begin_attempt(root, NOTEBOOK)
TRIP_PATH = root / manifest["inputs"]["trips"]
NETWORK_DIR = root / manifest["inputs"]["network"]

config = load_config(CONFIG)
if config.to_dict() != spec["config"]:
    raise ValueError("Experiment YAML differs from the frozen manifest; regenerate the experiment plan before running.")
display(pd.Series({{
    "Experiment": EXPERIMENT_ID,
    "Mode": spec["mode"],
    "Baseline": spec["baseline_id"],
    "Changed parameter": spec["parameter"],
    "Baseline value": spec["baseline_value"],
    "Experiment value": spec["value"],
}}, name="Setting").to_frame())
print(yaml.safe_dump(config.to_dict(), sort_keys=False))
'''),
            markdown(f"## Load inputs and prepare the simulation\n\n{prepare_note} The exact nearest-node lookup cache starts empty; the timed simulation includes its first lookups. The router builds its portal path cache in this cell. No result scoring is performed here."),
            code(prepare_code),
            markdown("## Run simulation — timed cell\n\nRun the preparation cell again before repeating this timer, so each measurement starts with a new simulator and router."),
            code('''started = perf_counter()
simulator.run()
simulation_seconds = perf_counter() - started
''', tags=["simulation", "timed"]),
            code('''matched_pairs = len(simulator.matches)
matched_travelers = 2 * matched_pairs
print(f"Simulation only: {simulation_seconds:.6f} seconds")
print(f"Matching size: {matched_pairs:,} carpool pairs / {matched_travelers:,} travelers")
print(f"Drive alone: {len(simulator.solos):,} travelers")
if len(cluster):
    print(f"Matched share: {matched_travelers / len(cluster):.2%}")
'''),
            markdown("## Save scored results and speed measurements\n\nThe five output files are `run.json`, `travelers.csv`, `matches.csv`, `legs.gpkg`, and `metrics.json`. Speed measurements use the elapsed time from the simulation cell only. Added time, waiting, and PNR access averages below are for matched travelers."),
            code('''tables = write_results(OUT_DIR, simulator, router, edges)
metrics = write_experiment_metrics(OUT_DIR, spec, simulator, simulation_seconds, tables=tables)
summary_fields = [
    "trip_count", "matched_pairs", "matched_travelers", "solo_travelers", "match_rate",
    "simulation_seconds", "travelers_per_second", "matched_pairs_per_second",
    "seconds_per_matched_pair", "vehicle_min_saved", "mean_delta_min",
    "mean_wait_min", "mean_access_min",
]
display(pd.Series({key: metrics[key] for key in summary_fields}, name="Result").to_frame())
roles = tables.travelers.groupby("role").agg(
    travelers=("trip_id", "size"),
    solo_min=("solo_min", "mean"),
    door_to_door_min=("carpool_min", "mean"),
    added_min=("delta_min", "mean"),
    wait_min=("wait_min", "mean"),
    access_min=("access_min", "mean"),
)
display(roles.round(2))
print(f"Saved to {OUT_DIR.relative_to(root)}")
'''),
            markdown("## Travel-time results\n\nThe first chart compares solo and simulated door-to-door travel times. The second shows the added-time distribution."),
            code('''before_after = plot_before_after(tables.travelers)
plt.show()
added_time = plot_delta(tables.travelers)
plt.show()
'''),
            markdown("## Individual trip\n\nSet `TRIP_ID` to inspect another traveler. The default selects a matched driver when available, otherwise a drive-alone traveler. Maps use saved road routes and work offline."),
            code('''TRIP_ID = None
if tables.travelers.empty:
    print("No trips are present in this input.")
else:
    selected_id = TRIP_ID
    if selected_id is None:
        selected_id = int(tables.matches.iloc[0]["driver_id"]) if len(tables.matches) else int(tables.travelers.iloc[0]["trip_id"])
    person = tables.travelers.loc[tables.travelers["trip_id"] == selected_id]
    if person.empty:
        raise ValueError(f"Trip {selected_id} is absent; choose a trip_id from travelers.csv.")
    display(person.round(2))
    partner = person.iloc[0]["partner_id"]
    if pd.notna(partner):
        display(tables.travelers.loc[tables.travelers["trip_id"] == int(partner)].round(2))
    trip_figure = plot_trip(selected_id, tables.travelers, tables.matches, tables.legs, basemap=False)
    plt.show()
'''),
            code('''complete_attempt(root, NOTEBOOK, execution_provenance)
print("Experiment evaluation complete; inputs, implementation, and notebook code fingerprints saved.")
'''),
        ]
    if joint:
        travel = next(
            index for index, cell in enumerate(cells)
            if cell.cell_type == "markdown" and cell.source.startswith("## Travel-time")
        )
        cells[travel:travel] = [
            markdown("## Pairs by mode\n\nDirect carpool runs first. Park-and-ride pairs are travelers the direct filters left unmatched."),
            code('''by_mode = tables.matches.groupby("mode").size().rename("pairs")
display(by_mode.to_frame())
'''),
        ]
    return notebook(cells)


def summary_notebook():
    return notebook(
        [
            markdown('''# Compare the experiment runs

The plan keeps the original ten direct-carpool runs, ten PNR runs, and one joint run (`e01` through `e21`). It then adds three levels for each sensitive control, so those controls have five values once the baseline and the original screen value are included. Horizon likewise has five levels: 5, 10, 15, 30, and 45 minutes. Both modes run with matching, bipartite, and greedy assignment. The Euclidean detour ratio has five levels: 1.05, 1.1, 1.25, 1.5, and 2. Each single-mode variant changes exactly one setting relative to its own mode's baseline. The joint run, `e21_both_direct_first`, keeps both baseline filter sets and runs direct carpool before park-and-ride. Every run uses the full Georgia Tech 2022 trip file and the same network and lot files.

Run the experiment notebooks first, or use `python scripts/run_experiments.py --all`. This overview reads saved outputs and never reruns a simulation. Missing, incomplete, or stale runs stay visible in the status table. Runtime includes only `simulator.run()`; data loading, scoring, exporting, and charts are excluded.

Each setting runs once. Runtime can vary with machine load and execution order; this is an initial parameter screen.
'''),
            code('''%matplotlib inline

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

here = Path.cwd().resolve()
for candidate in [here, *here.parents]:
    if (candidate / "evaluation").is_dir() and (candidate / "pyproject.toml").exists():
        root = candidate
        break
else:
    raise RuntimeError("Run this notebook from the CarpoolSim project or a folder inside it.")
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from carpoolsim.config import load_config
from evaluation.experiments import collect_experiment_metrics
from scripts.run_experiments import attempt_path, execution_fingerprint

manifest = json.loads((root / "config/experiments/manifest.json").read_text())
output_root = root / manifest["output_root"]
'''),
            markdown("## Completion and parameter sets\n\nA completed run has all five expected output files and the same configuration as the frozen manifest. The table always lists the complete plan."),
            code('''collected = collect_experiment_metrics(output_root)
records = {row["experiment_id"]: row for row in collected.to_dict("records")}
rows = []
required = ["run.json", "travelers.csv", "matches.csv", "legs.gpkg", "metrics.json"]
for spec in manifest["experiments"]:
    saved = records.get(spec["experiment_id"], {})
    folder = root / spec["output_path"]
    status = "missing"
    if any((folder / name).exists() for name in required):
        status = "incomplete"
    if saved and all((folder / name).is_file() for name in required):
        digest = hashlib.sha256(json.dumps(spec["config"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        same_config = (
            saved.get("config") == spec["config"]
            and saved.get("config_sha256") == digest
            and load_config(root / spec["config_path"]).to_dict() == spec["config"]
            and load_config(folder / "run.json").to_dict() == spec["config"]
        )
        status = "complete" if same_config and saved.get("status") == "complete" else "stale"
    notebook_path = root / spec["notebook_path"]
    attempt = json.loads(attempt_path(notebook_path).read_text()) if attempt_path(notebook_path).is_file() else {}
    current_fingerprint = execution_fingerprint(root, notebook_path)
    if status == "complete" and (
        saved.get("execution_fingerprint") != current_fingerprint
        or attempt.get("fingerprint") != current_fingerprint
        or attempt.get("status") != "complete"
    ):
        status = "stale"
    if attempt.get("status") == "running":
        status = "incomplete"
    elif attempt.get("status") == "failed" or notebook_path.with_name(notebook_path.stem + ".failed.ipynb").exists():
        status = "failed"
    row = {key: value for key, value in saved.items() if key != "config"}
    row.update({key: spec[key] for key in ["experiment_id", "mode", "baseline_id", "parameter", "baseline_value", "value", "description"]})
    row["status"] = status
    row["notebook_path"] = spec["notebook_path"]
    rows.append(row)

summary = pd.DataFrame(rows)
display(summary[["experiment_id", "mode", "parameter", "baseline_value", "value", "status"]])
print(f"Completed: {summary['status'].eq('complete').sum()} / {len(summary)} experiments")
completed = summary.loc[summary["status"] == "complete"].copy()
'''),
            markdown("## Simulation speed versus matching size\n\nMatching size is the number of committed carpool pairs, with two matched travelers per pair. The labels identify experiment numbers. Separate panels keep the modes visible without treating their costs as interchangeable."),
            code('''if completed.empty:
    print("No completed experiments yet. Run the experiment notebooks, then rerun this overview.")
else:
    colors = {"direct": "#2166ac", "pnr": "#b35806", "both": "#1a9850"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for ax, mode in zip(axes, ["direct", "pnr", "both"]):
        selected = completed.loc[completed["mode"] == mode]
        for _, row in selected.iterrows():
            baseline = row["parameter"] == "baseline"
            ax.scatter(row["matched_pairs"], row["simulation_seconds"], marker="*" if baseline else "o", s=160 if baseline else 55, color=colors[mode])
            ax.annotate(row["experiment_id"].split("_")[0], (row["matched_pairs"], row["simulation_seconds"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.set(title=f"{mode.upper()} — star marks baseline", xlabel="Matching size (carpool pairs)", ylabel="Simulation only (seconds)")
        ax.grid(alpha=0.2)
    plt.show()
'''),
            markdown("## Results relative to each mode's baseline\n\nDifferences subtract the corresponding direct, PNR, or joint baseline. The matched-share difference is in percentage points. Throughputs and seconds per matched pair are descriptive ratios; a zero matching size leaves seconds per pair undefined. When the three baselines are complete, the last table places the joint run beside the single-mode baselines and counts its pairs by mode."),
            code('''if not completed.empty:
    baseline_rows = completed.loc[completed["parameter"] == "baseline"].set_index("experiment_id")
    measures = ["matched_pairs", "simulation_seconds", "match_rate", "vehicle_min_saved", "mean_delta_min"]
    for measure in measures:
        reference = baseline_rows[measure] if measure in baseline_rows else pd.Series(dtype=float)
        completed[f"{measure}_difference"] = completed[measure] - completed["baseline_id"].map(reference)
    completed["match_rate_difference_pp"] = 100 * completed["match_rate_difference"]
    comparison_columns = [
        "experiment_id", "matched_pairs", "matched_pairs_difference", "match_rate",
        "match_rate_difference_pp", "simulation_seconds", "simulation_seconds_difference",
        "matched_pairs_per_second", "seconds_per_matched_pair", "vehicle_min_saved",
        "vehicle_min_saved_difference", "mean_delta_min", "mean_delta_min_difference",
    ]
    display(completed[comparison_columns].round(4))
    summary = summary.merge(
        completed[["experiment_id"] + [name for name in completed if name.endswith("_difference") or name.endswith("_difference_pp")]],
        on="experiment_id", how="left", validate="one_to_one",
    )

output_root.mkdir(parents=True, exist_ok=True)
summary.to_csv(output_root / "summary.csv", index=False)
print(f"Saved {len(summary)} planned runs to {(output_root / 'summary.csv').relative_to(root)}")

joint_id = "e21_both_direct_first"
reference_ids = ["e01_direct_baseline", "e11_pnr_baseline", joint_id]
if set(reference_ids) <= set(completed["experiment_id"]):
    side_by_side = completed.loc[completed["experiment_id"].isin(reference_ids), [
        "experiment_id", "mode", "matched_pairs", "matched_travelers", "solo_travelers",
        "match_rate", "vehicle_min_saved", "mean_delta_min", "mean_wait_min", "mean_access_min",
        "simulation_seconds",
    ]]
    display(side_by_side.round(4))
    joint_matches = pd.read_csv(root / f"data_outputs/experiments/{joint_id}/matches.csv")
    display(joint_matches.groupby("mode").size().rename("pairs").to_frame())
'''),
        ]
    )


def write_notebook(path: Path, generated, overwrite_executed: bool):
    if path.exists():
        existing = nbformat.read(path, as_version=4)
        original = [(cell.cell_type, cell.source) for cell in existing.cells]
        updated = [(cell.cell_type, cell.source) for cell in generated.cells]
        if original == updated:
            return
        executed = any(cell.get("execution_count") is not None for cell in existing.cells)
        if executed and not overwrite_executed:
            raise ValueError(f"Refusing to clear executed notebook {path}; use --overwrite-executed to regenerate it.")
    nbformat.validate(generated)
    nbformat.write(generated, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--overwrite-executed", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_dir = root / "config/experiments"
    notebook_dir = root / "notebooks/experiments"
    config_dir.mkdir(parents=True, exist_ok=True)
    notebook_dir.mkdir(parents=True, exist_ok=True)
    specs = experiment_specs(root)
    for spec in specs:
        experiment_id = spec["experiment_id"]
        spec.update(
            config_path=f"config/experiments/{experiment_id}.yaml",
            notebook_path=f"notebooks/experiments/{experiment_id}.ipynb",
            output_path=f"data_outputs/experiments/{experiment_id}",
        )
        write_notebook(root / spec["notebook_path"], experiment_notebook(spec), args.overwrite_executed)
        (root / spec["config_path"]).write_text(
            f"# {spec['description']}\n# Times are minutes; distances are kilometers; other numeric filters are ratios.\n"
            + yaml.safe_dump(spec["config"], sort_keys=False),
            encoding="utf-8",
        )
    write_notebook(notebook_dir / "summary.ipynb", summary_notebook(), args.overwrite_executed)
    manifest = {
        "schema_version": 1,
        "design": "one change per single-mode variant against a frozen mode-specific baseline; one joint baseline runs direct carpool before park-and-ride; sensitive controls, the horizon, and the Euclidean detour ratio have five levels; both modes use matching, bipartite, and greedy",
        "timing": "wall time of simulator.run() only; one fresh process, portal path cache, and exact nearest-node lookup cache per notebook",
        "repetitions": 1,
        "inputs": {"trips": "data_inputs/trip/gt_2022_off_campus_trips.csv", "network": "data_inputs/network", "lots": "data_inputs/network/lots.gpkg"},
        "output_root": "data_outputs/experiments",
        "summary_notebook": "notebooks/experiments/summary.ipynb",
        "experiments": specs,
    }
    (config_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["experiment_id", "mode", "baseline_id", "parameter", "baseline_value", "value", "description", "config_path", "notebook_path", "output_path"]
    with (config_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(specs)
    print(f"Generated {len(specs)} parameter sets and experiment notebooks, plus summary.ipynb.")


if __name__ == "__main__":
    main()
