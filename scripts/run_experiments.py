"""Execute experiment notebooks sequentially in fresh Python processes.

The in-process IPython kernel preserves ordinary notebook outputs without
requiring local Jupyter socket access. Each notebook still has an isolated
process, a fresh router, and its own simulation-only timer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "carpoolsim-matplotlib"))


def configuration_digest(config):
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def manifest(root):
    return json.loads((root / "config/experiments/manifest.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def environment_fingerprint(root):
    """Hash simulation/runner code and input contents once per worker process."""
    root = Path(root).resolve()
    files = []
    for package in ["carpoolsim", "evaluation"]:
        files.extend((root / package).rglob("*.py"))
    files.extend(root / "scripts" / name for name in ["create_experiments.py", "run_experiments.py"] if (root / "scripts" / name).is_file())
    inputs = manifest(root).get("inputs", {})
    if inputs:
        network = root / inputs["network"]
        files.extend([root / inputs["trips"], network / "nodes.gpkg", network / "edges.gpkg", root / inputs["lots"]])
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(block)
        digest.update(file_digest.digest())
    return digest.hexdigest()


def execution_fingerprint(root, notebook_path):
    """Bind results to input/code contents and the notebook's code cells."""
    import nbformat

    saved = nbformat.read(notebook_path, as_version=4)
    sources = [cell.source for cell in saved.cells if cell.cell_type == "code"]
    notebook_digest = hashlib.sha256(json.dumps(sources, ensure_ascii=False).encode()).hexdigest()
    return {"environment_sha256": environment_fingerprint(Path(root).resolve()), "notebook_code_sha256": notebook_digest}


def attempt_path(notebook_path):
    return Path(notebook_path).with_suffix(".attempt.json")


def set_attempt(notebook_path, status, fingerprint, **extra):
    """Persist the latest attempt independently of earlier successful outputs."""
    path = attempt_path(notebook_path)
    payload = {"status": status, "fingerprint": fingerprint, "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def begin_attempt(root, notebook_path):
    fingerprint = execution_fingerprint(root, notebook_path)
    set_attempt(notebook_path, "running", fingerprint, pid=os.getpid())
    return fingerprint


def complete_attempt(root, notebook_path, fingerprint):
    """Attach provenance to metrics after the notebook's evaluation succeeds."""
    spec = next((item for item in manifest(root)["experiments"] if (Path(root) / item["notebook_path"]).resolve() == Path(notebook_path).resolve()), None)
    if spec:
        path = Path(root) / spec["output_path"] / "metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metrics["execution_fingerprint"] = fingerprint
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    set_attempt(notebook_path, "complete", fingerprint)
    Path(notebook_path).with_name(Path(notebook_path).stem + ".failed.ipynb").unlink(missing_ok=True)


def write_atomic(notebook, path):
    import nbformat

    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        nbformat.validate(notebook)
        nbformat.write(notebook, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def completed_consistently(root, spec):
    """Resume only fully executed notebooks with matching configs and outputs."""
    import nbformat

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from carpoolsim.config import load_config

    output = root / spec["output_path"]
    notebook_path = root / spec["notebook_path"]
    if notebook_path.with_name(notebook_path.stem + ".failed.ipynb").exists():
        return False
    expected_files = ["run.json", "travelers.csv", "matches.csv", "legs.gpkg", "metrics.json"]
    if not all((output / name).is_file() for name in expected_files):
        return False
    try:
        saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        executed = nbformat.read(root / spec["notebook_path"], as_version=4)
        marker = executed.metadata.get("carpoolsim_execution", {})
        attempt = json.loads(attempt_path(notebook_path).read_text(encoding="utf-8"))
        fingerprint = execution_fingerprint(root, notebook_path)
        digest = configuration_digest(spec["config"])
        return (
            saved.get("status") == "complete"
            and saved.get("experiment_id") == spec["experiment_id"]
            and saved.get("config") == spec["config"]
            and saved.get("config_sha256") == digest
            and load_config(root / spec["config_path"]).to_dict() == spec["config"]
            and load_config(output / "run.json").to_dict() == spec["config"]
            and marker.get("status") == "complete"
            and marker.get("config_sha256") == digest
            and marker.get("fingerprint") == fingerprint
            and attempt.get("status") == "complete"
            and attempt.get("fingerprint") == fingerprint
            and saved.get("execution_fingerprint") == fingerprint
            and all(
                cell.get("execution_count") is not None
                and not any(item.output_type == "error" for item in cell.get("outputs", []))
                for cell in executed.cells
                if cell.cell_type == "code"
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def execute_notebook(root, path):
    """Worker entry point; run exactly one notebook and save its outputs."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "carpoolsim-matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("IPYTHONDIR", str(Path(tempfile.gettempdir()) / "carpoolsim-experiment-ipython"))
    os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["IPYTHONDIR"]).mkdir(parents=True, exist_ok=True)

    import nbformat
    from IPython.utils.capture import capture_output
    from ipykernel.inprocess.manager import InProcessKernelManager

    plan = manifest(root)
    spec = next(
        (item for item in plan["experiments"] if (root / item["notebook_path"]).resolve() == path),
        None,
    )
    saved = nbformat.read(path, as_version=4)
    saved.metadata.pop("carpoolsim_execution", None)
    for cell in saved.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    os.chdir(path.parent)
    fingerprint = begin_attempt(root, path)
    manager = InProcessKernelManager()
    manager.start_kernel()
    shell = manager.kernel.shell
    print(f"Running {path.relative_to(root)}", flush=True)
    try:
        for number, cell in enumerate(saved.cells, start=1):
            if cell.cell_type != "code":
                continue
            print(f"  Cell {number}/{len(saved.cells)}", flush=True)
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
            cell.execution_count = shell.execution_count - 1
            for name, content in [("stdout", captured.stdout), ("stderr", captured.stderr)]:
                if content:
                    cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=content))
                    print(content.rstrip(), flush=True)
            for output in captured.outputs:
                cell.outputs.append(
                    nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata)
                )
            error = result.error_before_exec or result.error_in_exec
            if error is not None:
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "error", ename=type(error).__name__, evalue=str(error),
                        traceback=traceback.format_exception(type(error), error, error.__traceback__),
                    )
                )
                saved.metadata["carpoolsim_execution"] = {"status": "failed", "failed_cell": number}
                failed_path = path.with_name(path.stem + ".failed.ipynb")
                write_atomic(saved, failed_path)
                set_attempt(path, "failed", fingerprint, failed_cell=number)
                raise RuntimeError(f"Cell {number} failed; partial outputs saved to {failed_path}") from error
        complete_attempt(root, path, fingerprint)
        marker = {"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "fingerprint": fingerprint}
        if spec:
            marker.update(experiment_id=spec["experiment_id"], config_sha256=configuration_digest(spec["config"]))
        saved.metadata["carpoolsim_execution"] = marker
        write_atomic(saved, path)
        path.with_name(path.stem + ".failed.ipynb").unlink(missing_ok=True)
        if spec:
            metrics = json.loads((root / spec["output_path"] / "metrics.json").read_text())
            print(
                f"Completed {spec['experiment_id']}: {metrics['matched_pairs']:,} pairs, "
                f"{metrics['simulation_seconds']:.6f}s simulation only.",
                flush=True,
            )
        else:
            print(f"Completed {path.name}", flush=True)
    finally:
        manager.shutdown_kernel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Run every experiment in order, then the summary.")
    selection.add_argument("--notebook", type=Path, help="Run one notebook; relative paths start at the project root.")
    selection.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--resume", action="store_true", help="Skip completed experiments whose configs and outputs still match.")
    args = parser.parse_args()
    root = args.root.resolve()
    if args._worker:
        execute_notebook(root, args._worker.resolve())
        return
    plan = manifest(root)
    if args.all:
        selected = [root / item["notebook_path"] for item in plan["experiments"]]
        selected.append(root / plan["summary_notebook"])
    else:
        path = args.notebook.expanduser()
        selected = [(path if path.is_absolute() else root / path).resolve()]
    specs = {(root / item["notebook_path"]).resolve(): item for item in plan["experiments"]}
    for position, notebook_path in enumerate(selected, start=1):
        notebook_path = notebook_path.resolve()
        if not notebook_path.is_file():
            parser.error(f"Notebook does not exist: {notebook_path}")
        spec = specs.get(notebook_path)
        if args.resume and spec and completed_consistently(root, spec):
            print(f"[{position}/{len(selected)}] Skip completed {spec['experiment_id']}", flush=True)
            continue
        print(f"[{position}/{len(selected)}] Start {notebook_path.name}", flush=True)
        subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--root", str(root), "--_worker", str(notebook_path)],
            check=True,
            cwd=root,
        )
    print("All requested notebooks completed.", flush=True)


if __name__ == "__main__":
    # Notebook cells import the same helpers, sharing the process-level cache.
    sys.modules.setdefault("scripts.run_experiments", sys.modules[__name__])
    main()
