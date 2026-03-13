#!/usr/bin/env python3
"""Benchmark runner for BindingMetrics metrics.

Runs each metric N times on every input in a dataset directory, collects
wall-clock timing statistics, and saves results as JSON + Markdown.

All metrics are discovered from ``binding_metrics.metrics.registry.METRICS``
— adding a metric to the registry automatically makes it appear here.

Usage:
    python benchmarks/run.py --dataset /path/to/benchmark/data
    python benchmarks/run.py --dataset /path/to/data --n-runs 20
    python benchmarks/run.py --dataset /path/to/data \\
        --input-types static_structure trajectory

Dataset directory layout:
    data/
        manifest.json          # required — see format below
        small.pdb
        medium.cif
        traj_short.dcd
        traj_short_top.pdb

manifest.json format:
    {
      "structures": [
        {
          "path": "small.pdb",
          "label": "small",
          "design_chain": "B"      // optional — auto-detected otherwise
        }
      ],
      "trajectories": [
        {
          "trajectory": "traj_short.dcd",
          "topology": "traj_short_top.pdb",
          "label": "short MD",
          "ligand_chain": "B",     // used to build ligand_indices
          "receptor_chain": "A"   // used to build receptor_indices
        }
      ]
    }

Results are written to benchmarks/results/<timestamp>.{json,md}.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from binding_metrics.metrics.registry import (
    METRICS,
    ChainMode,
    InputType,
    MetricSpec,
    metrics_by_input_type,
)

# MD parameters that can be overridden per-entry in the manifest "md" dict.
# Maps manifest key → RelaxationConfig attribute name (they're the same here).
_MD_PARAM_KEYS = (
    "md_duration_ps",
    "md_timestep_fs",
    "md_temperature_k",
    "md_friction",
    "md_save_interval_ps",
    "min_steps_initial",
    "min_steps_restrained",
    "min_steps_final",
    "min_tolerance",
    "restraint_strength",
    "solvent_model",
    "device",
    "peptide_chain_id",
    "receptor_chain_id",
)


# ---------------------------------------------------------------------------
# Structure property extraction (for scaling analysis)
# ---------------------------------------------------------------------------


def _load_biotite_atoms(path: Path):
    """Load PDB or CIF as a biotite AtomArray. Returns None on import error."""
    try:
        import biotite.structure.io.pdbx as pdbx
        import biotite.structure.io.pdb as pdb_io
    except ImportError:
        return None

    suffix = path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        f = pdbx.CIFFile.read(str(path))
        return pdbx.get_structure(f, model=1)
    else:
        f = pdb_io.PDBFile.read(str(path))
        return pdb_io.get_structure(f, model=1)


def compute_structure_properties(path: Path, design_chain: Optional[str] = None) -> dict:
    """Extract structural properties for scaling analysis.

    Returns n_atoms, n_residues_total, n_residues_peptide, n_residues_receptor,
    peptide_chain, receptor_chain.
    """
    import numpy as np

    props: dict = {
        "n_atoms": 0,
        "n_residues_total": 0,
        "n_residues_peptide": None,
        "n_residues_receptor": None,
        "peptide_chain": design_chain,
        "receptor_chain": None,
    }

    atoms = _load_biotite_atoms(path)
    if atoms is None:
        return props

    props["n_atoms"] = int(len(atoms))
    chains = np.unique(atoms.chain_id)
    props["n_residues_total"] = int(
        sum(len(np.unique(atoms[atoms.chain_id == c].res_id)) for c in chains)
    )

    try:
        from binding_metrics.metrics.interface import detect_interface_chains
        pep, rec = detect_interface_chains(atoms, design_chain)
        props["peptide_chain"] = pep
        props["receptor_chain"] = rec
        if pep is not None:
            props["n_residues_peptide"] = int(
                len(np.unique(atoms[atoms.chain_id == pep].res_id))
            )
        if rec is not None:
            props["n_residues_receptor"] = int(
                len(np.unique(atoms[atoms.chain_id == rec].res_id))
            )
    except Exception:
        pass

    return props


def compute_trajectory_properties(traj_path: Path, top_path: Path, entry: dict) -> dict:
    """Extract trajectory properties and resolve atom indices from chain IDs."""
    props: dict = {
        "n_frames": None,
        "n_atoms": None,
        "ligand_indices": entry.get("ligand_indices"),
        "receptor_indices": entry.get("receptor_indices"),
        "ligand_chain": entry.get("ligand_chain"),
        "receptor_chain": entry.get("receptor_chain"),
    }
    try:
        import mdtraj as md
        traj = md.load(str(traj_path), top=str(top_path))
        props["n_frames"] = traj.n_frames
        props["n_atoms"] = traj.n_atoms

        # Auto-resolve atom indices from chain IDs if not provided explicitly
        if props["ligand_indices"] is None and props["ligand_chain"]:
            sel = traj.topology.select(
                f"chainid {_chain_index(traj.topology, props['ligand_chain'])} "
                "and not type H"
            )
            props["ligand_indices"] = sel.tolist()

        if props["receptor_indices"] is None and props["receptor_chain"]:
            sel = traj.topology.select(
                f"chainid {_chain_index(traj.topology, props['receptor_chain'])} "
                "and not type H"
            )
            props["receptor_indices"] = sel.tolist()

    except Exception:
        pass
    return props


def _chain_index(topology, chain_id: str) -> int:
    """Return the MDTraj chain index for a given chain ID string."""
    for chain in topology.chains:
        cid = getattr(chain, "chain_id", None) or str(chain.index)
        if cid == chain_id:
            return chain.index
    raise ValueError(f"Chain {chain_id!r} not found in topology")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_calls(fn, *args, n_runs: int, **kwargs) -> list[float]:
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return times


def _stats(times: list[float]) -> dict:
    return {
        "mean_s": statistics.mean(times),
        "std_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_s": min(times),
        "max_s": max(times),
        "n_runs": len(times),
        "times_s": times,
    }


# ---------------------------------------------------------------------------
# Generic metric dispatcher (driven entirely by MetricSpec)
# ---------------------------------------------------------------------------


def _build_static_kwargs(spec: MetricSpec, path: Path, props: dict) -> Optional[dict]:
    """Build the kwargs dict to call a static_structure metric.

    Returns None if the metric cannot run on this input (e.g. format mismatch).
    """
    suffix = path.suffix.lower().lstrip(".")
    if spec.formats and suffix not in spec.formats:
        return None  # format not supported

    kwargs: dict = {spec.path_arg: path}

    if spec.secondary_path_arg:
        kwargs[spec.secondary_path_arg] = path  # same file = self-comparison

    mode: ChainMode = spec.chain_mode
    if mode == "single" and spec.chain_arg and props.get("peptide_chain"):
        kwargs[spec.chain_arg] = props["peptide_chain"]
    elif mode in ("interface", "interface_2paths"):
        if spec.peptide_chain_arg and props.get("peptide_chain"):
            kwargs[spec.peptide_chain_arg] = props["peptide_chain"]
        if spec.receptor_chain_arg and props.get("receptor_chain"):
            kwargs[spec.receptor_chain_arg] = props["receptor_chain"]

    return kwargs


def _build_trajectory_kwargs(
    spec: MetricSpec,
    traj_path: Path,
    top_path: Path,
    props: dict,
) -> Optional[dict]:
    """Build kwargs to call a trajectory metric from resolved props."""
    kwargs: dict = {spec.path_arg: traj_path}

    if "topology_path" in _fn_argnames(spec):
        kwargs["topology_path"] = top_path

    mode = spec.chain_mode
    if mode == "interface":
        # peptide_chain_arg / receptor_chain_arg store the kwarg names that the
        # function expects; both are resolved to atom index lists in props.
        if spec.peptide_chain_arg and props.get("ligand_indices") is not None:
            kwargs[spec.peptide_chain_arg] = props["ligand_indices"]
        if spec.receptor_chain_arg and props.get("receptor_indices") is not None:
            kwargs[spec.receptor_chain_arg] = props["receptor_indices"]
    elif mode == "single":
        # chain_arg receives the receptor chain ID string (compute_receptor_drift)
        if spec.chain_arg and props.get("receptor_chain"):
            kwargs[spec.chain_arg] = props["receptor_chain"]

    return kwargs


def _fn_argnames(spec: MetricSpec) -> set[str]:
    """Return the set of parameter names of the metric function."""
    import inspect
    try:
        fn = spec.load()
        return set(inspect.signature(fn).parameters)
    except Exception:
        return set()


def bench_static_metric(
    spec: MetricSpec, path: Path, props: dict, n_runs: int
) -> dict:
    kwargs = _build_static_kwargs(spec, path, props)
    if kwargs is None:
        return {"skipped": f"n/a ({'/'.join(spec.formats)} only)"}
    try:
        fn = spec.load()
        times = _time_calls(fn, n_runs=n_runs, **kwargs)
        return _stats(times)
    except Exception as exc:
        return {"error": str(exc)}


def bench_trajectory_metric(
    spec: MetricSpec, traj_path: Path, top_path: Path, props: dict, n_runs: int
) -> dict:
    kwargs = _build_trajectory_kwargs(spec, traj_path, top_path, props)
    if kwargs is None:
        return {"skipped": "missing required args"}
    try:
        fn = spec.load()
        times = _time_calls(fn, n_runs=n_runs, **kwargs)
        return _stats(times)
    except Exception as exc:
        return {"error": str(exc)}


def bench_md_simulation(path: Path, md_params: dict, output_dir: Path) -> dict:
    """Run one ImplicitRelaxation and return its internal timing.

    MD runs are expensive so they are never repeated — timing comes from
    RelaxationResult.minimization_time_s / .md_time_s, which are measured
    internally by the protocol.

    md_params: subset of RelaxationConfig fields from the manifest "md" key.
    """
    try:
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        config = RelaxationConfig(**{k: v for k, v in md_params.items() if k in _MD_PARAM_KEYS})
        relaxer = ImplicitRelaxation(config)
        result = relaxer.run(path, output_dir / path.stem)

        if not result.success:
            return {"error": result.error_message or "unknown error"}

        return {
            "minimization_time_s": result.minimization_time_s,
            "md_time_s": result.md_time_s,
            "total_time_s": (result.minimization_time_s or 0) + (result.md_time_s or 0),
            "md_params": md_params,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_manifest(dataset_dir: Path) -> dict:
    """Load manifest.json from dataset_dir. Returns {"structures": [], "trajectories": []}."""
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json found in {dataset_dir}.\n"
            "Create one listing your structures and/or trajectories. "
            "See benchmarks/manifest_example.json for the format."
        )
    with open(manifest_path) as f:
        data = json.load(f)

    # Resolve all paths relative to the manifest directory
    for entry in data.get("structures", []):
        entry["path"] = str(dataset_dir / entry["path"])
    for entry in data.get("trajectories", []):
        entry["trajectory"] = str(dataset_dir / entry["trajectory"])
        entry["topology"] = str(dataset_dir / entry["topology"])
    for entry in data.get("simulations", []):
        entry["path"] = str(dataset_dir / entry["path"])

    return data


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def run_static_benchmarks(
    entries: list[dict], specs: list[MetricSpec], n_runs: int
) -> list[dict]:
    results = []
    for entry in entries:
        path = Path(entry["path"])
        label = entry.get("label", path.stem)
        design_chain = entry.get("design_chain")

        print(f"\n[{label}]  {path.name}")
        props = compute_structure_properties(path, design_chain)
        print(
            f"  atoms={props['n_atoms']}  "
            f"res_total={props['n_residues_total']}  "
            f"peptide={props.get('peptide_chain')}({props.get('n_residues_peptide')})  "
            f"receptor={props.get('receptor_chain')}({props.get('n_residues_receptor')})"
        )

        record: dict = {
            "label": label,
            "path": str(path),
            "input_type": "static_structure",
            "properties": props,
            "timings": {},
        }

        for spec in specs:
            print(f"  {spec.name:<26}", end=" ", flush=True)
            timing = bench_static_metric(spec, path, props, n_runs)
            record["timings"][spec.name] = timing
            print(_fmt_timing(timing))

        results.append(record)
    return results


def run_trajectory_benchmarks(
    entries: list[dict], specs: list[MetricSpec], n_runs: int
) -> list[dict]:
    results = []
    for entry in entries:
        traj_path = Path(entry["trajectory"])
        top_path = Path(entry["topology"])
        label = entry.get("label", traj_path.stem)

        print(f"\n[{label}]  {traj_path.name}")
        props = compute_trajectory_properties(traj_path, top_path, entry)
        print(
            f"  frames={props.get('n_frames')}  atoms={props.get('n_atoms')}  "
            f"ligand_chain={props.get('ligand_chain')} "
            f"({len(props['ligand_indices']) if props.get('ligand_indices') else '?'} atoms)  "
            f"receptor_chain={props.get('receptor_chain')} "
            f"({len(props['receptor_indices']) if props.get('receptor_indices') else '?'} atoms)"
        )

        record: dict = {
            "label": label,
            "trajectory": str(traj_path),
            "topology": str(top_path),
            "input_type": "trajectory",
            "properties": props,
            "timings": {},
        }

        for spec in specs:
            print(f"  {spec.name:<26}", end=" ", flush=True)
            timing = bench_trajectory_metric(spec, traj_path, top_path, props, n_runs)
            record["timings"][spec.name] = timing
            print(_fmt_timing(timing))

        results.append(record)
    return results


def run_md_benchmarks(entries: list[dict], output_dir: Path) -> list[dict]:
    """Run implicit MD relaxation for each simulation entry.

    MD runs are not repeated (n_runs has no effect here). Timing is read
    directly from RelaxationResult internal fields.
    """
    results = []
    for entry in entries:
        path = Path(entry["path"])
        label = entry.get("label", path.stem)
        md_params = entry.get("md", {})

        print(f"\n[{label}]  {path.name}")
        props = compute_structure_properties(path, entry.get("design_chain"))
        print(
            f"  atoms={props['n_atoms']}  res_total={props['n_residues_total']}  "
            f"md_duration_ps={md_params.get('md_duration_ps', 200)}  "
            f"device={md_params.get('device', 'cuda')}"
        )

        record: dict = {
            "label": label,
            "path": str(path),
            "input_type": "md_simulation",
            "properties": props,
            "md_params": md_params,
            "timings": {},
        }

        print(f"  {'md_implicit':<26}", end=" ", flush=True)
        timing = bench_md_simulation(path, md_params, output_dir / "md_outputs")
        record["timings"]["md_implicit"] = timing
        if "error" in timing:
            print(f"error: {timing['error'][:60]}")
        else:
            min_s = timing.get("minimization_time_s") or 0
            md_s = timing.get("md_time_s") or 0
            print(f"min={min_s*1000:.0f} ms  md={md_s*1000:.0f} ms  total={timing['total_time_s']*1000:.0f} ms")

        results.append(record)
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_STATIC_PROP_COLS = [
    "n_atoms", "n_residues_total", "n_residues_peptide", "n_residues_receptor"
]
_TRAJ_PROP_COLS = ["n_frames", "n_atoms"]
_MD_PROP_COLS = ["n_atoms", "n_residues_total"]


def _fmt_timing(t: dict) -> str:
    if "skipped" in t:
        return t["skipped"]
    if "error" in t:
        return f"error: {t['error'][:60]}"
    # MD timing (single run, not repeated)
    if "total_time_s" in t:
        min_s = t.get("minimization_time_s") or 0
        md_s = t.get("md_time_s") or 0
        return f"min={min_s*1000:.0f}ms md={md_s*1000:.0f}ms"
    if "mean_s" in t:
        return f"{t['mean_s'] * 1000:.1f} ± {t['std_s'] * 1000:.1f} ms"
    return "—"


def _markdown_table(
    results: list[dict], prop_cols: list[str], metric_names: list[str]
) -> str:
    cols = ["input"] + prop_cols + metric_names
    rows = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for r in results:
        props = r["properties"]
        row = [r["label"]]
        for p in prop_cols:
            v = props.get(p)
            row.append(str(v) if v is not None else "—")
        for m in metric_names:
            row.append(_fmt_timing(r["timings"].get(m, {})))
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def format_report(
    static_results: list[dict],
    traj_results: list[dict],
    md_results: list[dict],
    static_specs: list[MetricSpec],
    traj_specs: list[MetricSpec],
) -> str:
    parts = []
    if static_results:
        parts.append("## Static structure metrics\n")
        parts.append(_markdown_table(
            static_results, _STATIC_PROP_COLS, [s.name for s in static_specs]
        ))
    if traj_results:
        parts.append("\n\n## Trajectory metrics\n")
        parts.append(_markdown_table(
            traj_results, _TRAJ_PROP_COLS, [s.name for s in traj_specs]
        ))
    if md_results:
        parts.append("\n\n## MD simulation (ImplicitRelaxation)\n")
        parts.append(_markdown_table(
            md_results, _MD_PROP_COLS, ["md_implicit"]
        ))
    return "\n".join(parts)


def _json_default(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_results(
    static_results: list[dict],
    traj_results: list[dict],
    md_results: list[dict],
    static_specs: list[MetricSpec],
    traj_specs: list[MetricSpec],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    payload = {
        "timestamp": timestamp,
        "static_metrics": [s.name for s in static_specs],
        "trajectory_metrics": [s.name for s in traj_specs],
        "md_metrics": ["md_implicit"] if md_results else [],
        "static_results": static_results,
        "trajectory_results": traj_results,
        "md_results": md_results,
    }
    json_path = output_dir / f"{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    report_path = output_dir / f"{timestamp}.md"
    report_path.write_text(
        format_report(static_results, traj_results, md_results, static_specs, traj_specs) + "\n"
    )

    return json_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_ALL_METRIC_NAMES = [m.name for m in METRICS]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BindingMetrics metrics (driven by the metric registry)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="Directory containing a manifest.json and the benchmark inputs",
    )
    parser.add_argument(
        "--n-runs", type=int, default=10,
        help="Number of timed runs per metric per input (default: 10)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results"),
        help="Directory for result files (default: benchmarks/results)",
    )
    parser.add_argument(
        "--input-types", nargs="+",
        default=["static_structure", "trajectory", "md_simulation"],
        choices=["static_structure", "trajectory", "md_simulation"],
        metavar="TYPE",
        help="Input types to benchmark (default: all three)",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=None,
        choices=_ALL_METRIC_NAMES,
        metavar="METRIC",
        help=(
            "Restrict to specific metrics. "
            f"All available: {', '.join(_ALL_METRIC_NAMES)}"
        ),
    )
    args = parser.parse_args()

    manifest = load_manifest(args.dataset)

    # Select specs matching requested input types and optional metric filter
    def _select(input_type: InputType) -> list[MetricSpec]:
        specs = metrics_by_input_type(input_type)
        if args.metrics:
            specs = [s for s in specs if s.name in args.metrics]
        return specs

    static_specs = _select("static_structure") if "static_structure" in args.input_types else []
    traj_specs = _select("trajectory") if "trajectory" in args.input_types else []
    run_md = "md_simulation" in args.input_types

    static_entries = manifest.get("structures", [])
    traj_entries = manifest.get("trajectories", [])
    sim_entries = manifest.get("simulations", [])

    n_inputs = len(static_entries) + len(traj_entries) + len(sim_entries)
    if n_inputs == 0:
        print("No inputs found in manifest.json.")
        sys.exit(1)

    print(
        f"Benchmarking  structures={len(static_entries)}  "
        f"trajectories={len(traj_entries)}  simulations={len(sim_entries)}  "
        f"× {args.n_runs} runs"
    )

    static_results, traj_results, md_results = [], [], []

    if static_specs and static_entries:
        print("\n=== Static structure metrics ===")
        static_results = run_static_benchmarks(static_entries, static_specs, args.n_runs)

    if traj_specs and traj_entries:
        print("\n=== Trajectory metrics ===")
        traj_results = run_trajectory_benchmarks(traj_entries, traj_specs, args.n_runs)

    if run_md and sim_entries:
        print("\n=== MD simulation (ImplicitRelaxation) ===")
        md_results = run_md_benchmarks(sim_entries, args.output)

    report = format_report(static_results, traj_results, md_results, static_specs, traj_specs)
    print("\n" + report)

    json_path = save_results(
        static_results, traj_results, md_results, static_specs, traj_specs, args.output
    )
    print(f"\nSaved: {json_path}")
    print(f"Saved: {json_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
