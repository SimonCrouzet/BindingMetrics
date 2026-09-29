"""Full binding-metrics pipeline: prep → relax → energy → interface → geometry → electrostatics.

Runs every analysis step on a single structure and writes results to JSON.

Usage:
    binding-metrics-run \\
        --input complex.cif \\
        --output-dir results/ \\
        [--skip-prep] [--skip-relax] \\
        [--ph 7.4] \\
        [--metrics energy,interface,geometry,electrostatics,openfold] \\
        [--md-duration-ps 200] \\
        [--random-seed 1] \\
        [--device cuda]

The results JSON carries a ``provenance`` block (package version, git sha, seed,
platform) so a result can be tied to the code and settings that produced it.

Configuration file:
    --config run.toml supplies option defaults; flags on the command line override
    the file. Keys are the long option names (md-duration-ps or md_duration_ps):

        # run.toml
        md-duration-ps = 100
        ph = 7.0
        metrics = "interface,geometry"
        energy-modes = ["relaxed", "raw"]
        skip-prep = true

    A flag takes true or false, an option with several values takes a list, and
    an unknown key is an error.
"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Sequence

from binding_metrics._constants import (
    DEFAULT_DEVICE,
    DEFAULT_MD_DURATION_PS,
    DEFAULT_PH,
    DEFAULT_RANDOM_SEED,
)
from binding_metrics.cli import (
    add_config_arg,
    add_openfold_seeds_arg,
    md_save_interval_for,
    parse_args_with_config,
)
from binding_metrics.cli import seed_arg as _seed_arg
from binding_metrics.metrics._common import ChainAliasAction, resolve_chain_role
from binding_metrics.metrics.registry import get_metric
from binding_metrics.protocols.relaxer import Relaxer
from binding_metrics.provenance import collect_provenance
from binding_metrics.utils import configure_logging

# Named explicitly: ``python -m binding_metrics.cli.run`` executes this file as
# ``__main__``, and a logger called ``__main__`` would sit outside the package
# logger that ``configure_logging`` sets up, so its INFO lines would be lost.
logger = logging.getLogger("binding_metrics.cli.run")

#: Registry input types whose functions take an in-memory object (a loaded
#: ``AtomArray``, or a model's confidence arrays) and not a structure path, so a
#: pipeline step cannot call them.
_NON_PATH_INPUT_TYPES = frozenset({"atom_array", "predicted_structure"})

#: Pipeline step (the name ``--metrics`` takes) -> registry metrics the step runs.
#: The names differ from the registry's where one step calls several functions
#: (``geometry``) or the registry name says more (``structure_interaction_energy``
#: is the per-structure energy, ``interaction_energy`` the per-frame one).
_STEP_METRICS = {
    "energy": ("structure_interaction_energy",),
    "interface": ("interface",),
    "geometry": ("ramachandran", "omega", "shape_complementarity"),
    "electrostatics": ("coulomb",),
    "openfold": ("openfold",),
}


def _steps_taking_a_path(step_metrics: dict) -> frozenset:
    """Steps of ``step_metrics`` whose registry metrics all read a structure path.

    ``get_metric`` raises ``KeyError`` for a name the registry lacks, so a step
    cannot silently outlive the metric it runs.
    """
    return frozenset(
        step
        for step, names in step_metrics.items()
        if all(get_metric(name).input_type not in _NON_PATH_INPUT_TYPES for name in names)
    )


ALL_METRICS = _steps_taking_a_path(_STEP_METRICS)
# Reference-based metrics require a native structure (--reference) and are not
# part of the default set; they are auto-enabled when a reference is supplied.
REFERENCE_METRICS = _steps_taking_a_path({"dockq": ("dockq",)})
KNOWN_METRICS = ALL_METRICS | REFERENCE_METRICS


class ChainNotFoundError(ValueError):
    """A chain ID requested by the caller does not exist in the structure."""


def _require_chains_present(
    chain_info: dict, peptide_chain: Optional[str], receptor_chain: Optional[str]
) -> None:
    """Raise if an explicitly requested chain ID is absent from the structure.

    ``detect_chains_from_file`` echoes explicit IDs back without checking them,
    so a typo (``--peptide-chain Z``) would otherwise surface much later as an
    empty selection or NaN in every step. Only IDs the caller passed are
    checked; auto-detected ones come from ``all_chains`` by construction.
    ``all_chains`` lists amino-acid chains, so a ligand-only chain counts as
    absent.

    Raises:
        ChainNotFoundError: (a ``ValueError``) naming the missing chain(s) and
            listing the available ones as ``"B (13), A (85)"`` (id and residue
            count, smallest chain first).
    """
    available = chain_info["all_chains"]
    known = {c["id"] for c in available}
    missing = [c for c in (peptide_chain, receptor_chain) if c is not None and c not in known]
    if not missing:
        return
    listing = ", ".join(f"{c['id']} ({c['n_residues']})" for c in available)
    if len(missing) == 1:
        named = f"chain {missing[0]!r}"
    else:
        named = "chains " + ", ".join(repr(c) for c in missing)
    raise ChainNotFoundError(f"{named} not found; available: {listing}")


def _merge_reason(target: dict, extra: dict, label: str) -> None:
    """Move ``extra["reason"]`` into ``target["reason"]`` as ``"<label>: <reason>"``.

    Metric dicts merged into one flat namespace (OpenFold, then the EvoBind
    metrics that reuse its output) each carry an optional ``reason``; a plain
    ``dict.update`` would let the last one erase the diagnosis of the first.
    Reasons are joined with ``"; "``, and nothing is added when ``extra`` has none.
    """
    reason = extra.pop("reason", None)
    if reason:
        target["reason"] = "; ".join(filter(None, [target.get("reason"), f"{label}: {reason}"]))


def _warn(msg: str) -> None:
    logger.warning("  [warning] %s", msg)


def _step(name: str) -> None:
    bar = "=" * 60
    logger.info("\n%s\n  Step: %s\n%s", bar, name, bar)


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    sample_id: Optional[str] = None,
    # prep
    skip_prep: bool = False,
    ph: float = DEFAULT_PH,
    keep_water: bool = False,
    canonicalize: bool = False,
    # relax
    skip_relax: bool = False,
    md_duration_ps: float = DEFAULT_MD_DURATION_PS,
    device: str = DEFAULT_DEVICE,
    peptide_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    # metrics
    metrics: frozenset = ALL_METRICS,
    energy_modes: tuple = ("relaxed",),
    # reference-based (dockq)
    reference_path: Optional[Path] = None,
    # openfold
    openfold_mode: str = "score",
    openfold_conda_env: Optional[str] = None,
    # reproducibility
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    openfold_seeds: Optional[Sequence[int]] = None,
    *,
    binder_chain: Optional[str] = None,
    target_chain: Optional[str] = None,
    relaxer: Optional[Relaxer] = None,
) -> dict:
    """Run the full pipeline and return a results dict.

    Args:
        input_path: Complex structure (CIF or PDB).
        output_dir: Directory for intermediate files; created when missing.
        peptide_chain, receptor_chain: Explicit chain IDs (auth IDs). Auto-detected
            when None; an ID that is not in the structure raises ``ChainNotFoundError``.
        binder_chain, target_chain: Aliases of ``peptide_chain`` and ``receptor_chain``
            (keyword-only). Both spellings with different IDs raise ``ValueError``.
        relaxer: A ``Relaxer`` to run in place of the default ``ImplicitRelaxation``
            (keyword-only). It carries its own configuration, so ``md_duration_ps``,
            ``device`` and the chain IDs are not passed to it; ``random_seed`` still
            seeds prep and the energy step. Ignored when ``skip_relax`` is true.
        random_seed: Seed for hydrogen placement and MD; ``None`` for fresh randomness.
        openfold_seeds: Seed values written to the OpenFold3 query JSON; ``None``
            keeps the OpenFold default. Separate from ``random_seed``.
        The remaining arguments mirror the ``binding-metrics-run`` flags.

    Returns:
        Dict with ``sample_id``, ``input``, ``provenance`` (see
        ``binding_metrics.provenance.collect_provenance``), ``chains``, ``prep``,
        ``relax``, and one entry per metric (``energy``, ``interface``,
        ``geometry``, ``electrostatics``, ``dockq``, ``openfold``). A metric that
        did not run is ``{"skipped": True}``; one that failed is
        ``{"error": message}``.

        ``prep`` records what preparation changed: ``removed_heterogens``,
        ``n_removed_waters``, ``kept_nonstandard``, ``n_missing_atoms_rebuilt``
        and ``n_missing_residue_gaps`` (see ``core.system.prep_structure``).

    Raises:
        ChainNotFoundError: a requested chain ID does not exist in the structure.
        ValueError: a chain is given through both spellings with different IDs.
    """
    peptide_chain = resolve_chain_role("peptide_chain", peptide_chain, "binder_chain", binder_chain)
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain
    )
    if sample_id is None:
        sample_id = input_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "sample_id": sample_id,
        "input": str(input_path),
        "provenance": collect_provenance(seed=random_seed),
    }

    # ---------------------------------------------------------- Chain detection
    from binding_metrics.io.structures import detect_chains_from_file

    chain_info = detect_chains_from_file(
        input_path,
        peptide_chain=peptide_chain,
        receptor_chain=receptor_chain,
        verbose=True,
    )
    _require_chains_present(chain_info, peptide_chain, receptor_chain)
    peptide_chain = chain_info["peptide_chain"]  # auth_asym_id (biotite)
    receptor_chain = chain_info["receptor_chain"]
    peptide_chain_label = chain_info["peptide_chain_label"]  # label_asym_id (OpenMM)
    receptor_chain_label = chain_info["receptor_chain_label"]
    results["chains"] = chain_info

    # ------------------------------------------------------------------- Prep
    prepped_path = input_path
    if not skip_prep:
        _step("Structure Preparation (PDBFixer)")
        try:
            from binding_metrics.core.system import HAS_PDBFIXER, prep_structure
            from binding_metrics.io.structures import load_structure, save_structure

            if not HAS_PDBFIXER:
                _warn(
                    "pdbfixer not available — skipping prep. Install with: "
                    "pip install binding-metrics[structure]"
                )
            else:
                topology, positions = load_structure(input_path)
                prep_report: dict = {}
                topology, positions = prep_structure(
                    topology,
                    positions,
                    ph=ph,
                    keep_water=keep_water,
                    canonicalize=canonicalize,
                    random_seed=random_seed,
                    report=prep_report,
                )
                prepped_path = output_dir / f"{sample_id}_cleaned.cif"
                save_structure(topology, positions, prepped_path, source_path=input_path)
                logger.info("  Prepped structure: %s", prepped_path)
                results["prep"] = {
                    "output": str(prepped_path),
                    "ph": ph,
                    "keep_water": keep_water,
                    **prep_report,
                }
                # save_cif preserves original auth IDs and aligns label IDs to match,
                # so downstream OpenMM steps will see the original chain IDs.
                # Re-detect from the cleaned file so peptide_chain_label is up-to-date.
                prepped_chain_info = detect_chains_from_file(
                    prepped_path,
                    peptide_chain=peptide_chain,
                    receptor_chain=receptor_chain,
                )
                peptide_chain_label = prepped_chain_info["peptide_chain_label"]
                receptor_chain_label = prepped_chain_info["receptor_chain_label"]
        except Exception as e:
            _warn(f"Prep failed: {e} — continuing with raw input")
            traceback.print_exc()
            prepped_path = input_path
            results["prep"] = {"error": str(e)}
    else:
        logger.info("\n  [skip] Prep skipped — using raw input.")
        results["prep"] = {"skipped": True}

    # ------------------------------------------------------------------ Relax
    # Detect cyclic bonds from the original file before PDBFixer strips STRUCT_CONN.
    # Passed as hints to relaxation so cyclization survives the prep round-trip.
    cyclic_bond_hints = []
    try:
        from binding_metrics.core.cyclic import detect_cyclization
        from binding_metrics.io.structures import load_structure

        _orig_topo, _orig_pos = load_structure(input_path)
        cyclic_bond_hints = detect_cyclization(_orig_topo, _orig_pos, peptide_chain_label)
        if cyclic_bond_hints:
            logger.info(
                "  Cyclic bond hints from original file: %s",
                [b.cyclic_type for b in cyclic_bond_hints],
            )
    except Exception:
        # The hints are best effort. Without them relaxation detects cyclisation
        # from the prepped file, which may have lost the STRUCT_CONN records.
        pass

    relaxed_path: Optional[Path] = None
    if not skip_relax:
        _step("Relaxation (implicit MD)")
        if relaxer is None:
            if device == "cpu" and md_duration_ps > 0:
                logger.warning(
                    "\n  *** WARNING: running MD on CPU is extremely slow "
                    "and not recommended. ***\n"
                    "  *** For production use, run on a CUDA-capable GPU (--device cuda).   ***\n"
                    "  *** Use --md-duration-ps 0 to minimize only if GPU is unavailable.   ***\n"
                )
            from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

            config = RelaxationConfig(
                md_duration_ps=md_duration_ps,
                md_save_interval_ps=md_save_interval_for(md_duration_ps),
                device=device,
                peptide_chain_id=peptide_chain_label,
                receptor_chain_id=receptor_chain_label,
                cyclic_bond_hints=cyclic_bond_hints or None,
                # Auto-parameterise any non-canonical residue (e.g. cyclosporin's
                # BMT/ABA) with GAFF2 ExternalBond templates so relaxation builds.
                small_molecules="auto",
                random_seed=random_seed,
            )
            relaxer = ImplicitRelaxation(config)
        t0 = time.time()
        relax_result = relaxer.run(prepped_path, output_dir, sample_id=sample_id)
        elapsed = time.time() - t0

        results["relax"] = relax_result.to_dict()
        results["relax"]["elapsed_s"] = round(elapsed, 1)
        if results["relax"].get("qc_passed") is False:
            # Advisory only: the QC verdict never makes the run exit non-zero.
            logger.warning(
                "\n[WARNING] Structural QC failed: %s", results["relax"]["qc_failed_checks"]
            )
        # The platform OpenMM actually ran on, when the relaxation reports it.
        results["provenance"]["platform"] = results["relax"].get("platform")

        if not relax_result.success:
            logger.warning("\n[FAILED] Relaxation failed: %s", relax_result.error_message)
            logger.info("  Continuing with prepped input for downstream steps...")
            relaxed_path = prepped_path
            working_peptide = peptide_chain_label
            working_receptor = receptor_chain_label
        else:
            # Prefer MD-final structure; fall back to minimized
            if relax_result.md_final_structure_path:
                relaxed_path = Path(relax_result.md_final_structure_path)
            else:
                relaxed_path = Path(relax_result.minimized_structure_path)
            logger.info("\n  Relaxed structure: %s", relaxed_path)
            # OpenMM writes the relaxed CIF using label IDs as both auth and label,
            # so all downstream steps should use the label IDs.
            working_peptide = peptide_chain_label
            working_receptor = receptor_chain_label
    else:
        # Prefer PDBFixer-prepped structure (proper termini, removed heterogens)
        # over raw input; fall back to raw only if prep was skipped or failed.
        relaxed_path = prepped_path if prepped_path != input_path else input_path
        working_peptide = peptide_chain_label
        working_receptor = receptor_chain_label
        logger.info(
            "\n  [skip] Relaxation skipped — using %s input for downstream steps.",
            "prepped" if relaxed_path != input_path else "raw",
        )
        results["relax"] = {"skipped": True}

    # ------------------------------------------------------------------ Energy
    if "energy" in metrics:
        _step("Interaction Energy")
        try:
            from binding_metrics.metrics.energy import compute_interaction_energy

            energy = compute_interaction_energy(
                relaxed_path,
                peptide_chain=peptide_chain_label,
                receptor_chain=receptor_chain_label,
                device=device,
                sample_id=sample_id,
                modes=energy_modes,
                ph=ph,
                random_seed=random_seed,
            )
            results["energy"] = energy
        except Exception as e:
            _warn(f"Energy computation failed: {e}")
            traceback.print_exc()
            results["energy"] = {"error": str(e)}
    else:
        results["energy"] = {"skipped": True}

    # --------------------------------------------------------------- Interface
    if "interface" in metrics:
        _step("Interface Metrics (SASA, H-bonds, salt bridges)")
        try:
            from binding_metrics.metrics.interface import compute_interface_metrics

            interface = compute_interface_metrics(
                relaxed_path,
                design_chain=working_peptide,
                receptor_chain=working_receptor,
            )
            results["interface"] = interface
        except Exception as e:
            _warn(f"Interface metrics failed: {e}")
            traceback.print_exc()
            results["interface"] = {"error": str(e)}
    else:
        results["interface"] = {"skipped": True}

    # --------------------------------------------------------------- Geometry
    if "geometry" in metrics:
        _step("Geometry (Ramachandran + omega planarity + shape complementarity)")
        try:
            from binding_metrics.metrics.geometry import (
                compute_omega_planarity,
                compute_ramachandran,
                compute_shape_complementarity,
            )

            rama = compute_ramachandran(relaxed_path, chain=working_peptide)
            omega = compute_omega_planarity(relaxed_path, chain=working_peptide)
            sc = compute_shape_complementarity(
                relaxed_path,
                peptide_chain=working_peptide,
                receptor_chain=working_receptor,
            )
            results["geometry"] = {
                "ramachandran": rama,
                "omega": omega,
                "shape_complementarity": sc,
            }
        except Exception as e:
            _warn(f"Geometry metrics failed: {e}")
            traceback.print_exc()
            results["geometry"] = {"error": str(e)}
    else:
        results["geometry"] = {"skipped": True}

    # --------------------------------------------------------- Electrostatics
    if "electrostatics" in metrics:
        _step("Electrostatics (Coulomb cross-chain)")
        try:
            from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain

            elec = compute_coulomb_cross_chain(
                relaxed_path,
                peptide_chain=working_peptide,
                receptor_chain=working_receptor,
            )
            results["electrostatics"] = elec
        except Exception as e:
            _warn(f"Electrostatics failed: {e}")
            traceback.print_exc()
            results["electrostatics"] = {"error": str(e)}
    else:
        results["electrostatics"] = {"skipped": True}

    # ----------------------------------------------- DockQ (reference-based)
    if "dockq" in metrics:
        _step("DockQ CAPRI accuracy (vs reference)")
        if reference_path is None:
            _warn("DockQ requested but no reference structure was provided for this run; skipping.")
            results["dockq"] = {"skipped": True}
        else:
            try:
                from binding_metrics.metrics.dockq import compute_dockq_metrics

                # Score the prediction *as submitted* (original input coordinates)
                # against the reference: CAPRI evaluates the predicted coordinates,
                # so we deliberately use input_path, not the prepped/relaxed pose
                # (prep + MD would move atoms and confound the accuracy measure).
                dockq = compute_dockq_metrics(input_path, reference_path)
                results["dockq"] = dockq
                score = dockq.get("dockq")
                if score is not None:
                    logger.info(f"  DockQ: {score:.3f} ({dockq.get('capri_class')})")
            except Exception as e:
                _warn(f"DockQ failed: {e}")
                traceback.print_exc()
                results["dockq"] = {"error": str(e)}
    else:
        results["dockq"] = {"skipped": True}

    # --------------------------------------------------------- OpenFold
    if "openfold" in metrics:
        _step("OpenFold3 confidence scoring")
        try:
            from binding_metrics.metrics.openfold import (
                compute_openfold_metrics,
                run_openfold_refolding,
                run_openfold_scoring,
            )

            if not peptide_chain or not receptor_chain:
                _warn(
                    "OpenFold requires --peptide-chain and --receptor-chain "
                    "(or auto-detect); skipping."
                )
                results["openfold"] = {"skipped": True}
            else:
                of_dir = output_dir / "openfold"
                seed_kwargs = {"seeds": tuple(openfold_seeds)} if openfold_seeds else {}
                if openfold_mode == "refold":
                    predictions_dir = run_openfold_refolding(
                        complex_structure_path=input_path,
                        receptor_chain=receptor_chain,
                        binder_chain=peptide_chain,
                        query_name=sample_id,
                        output_dir=of_dir,
                        conda_env=openfold_conda_env,
                        **seed_kwargs,
                    )
                    of_metrics = compute_openfold_metrics(
                        output_dir=predictions_dir,
                        query_name=sample_id,
                        binder_chain=peptide_chain,
                        receptor_chain=receptor_chain,
                        reference_structure_path=input_path,
                    )
                else:  # score (default)
                    predictions_dir = run_openfold_scoring(
                        complex_structure_path=input_path,
                        receptor_chain=receptor_chain,
                        binder_chain=peptide_chain,
                        query_name=sample_id,
                        output_dir=of_dir,
                        conda_env=openfold_conda_env,
                        **seed_kwargs,
                    )
                    of_metrics = compute_openfold_metrics(
                        output_dir=predictions_dir,
                        query_name=sample_id,
                        binder_chain=peptide_chain,
                        receptor_chain=receptor_chain,
                    )
                # EvoBind metrics — no extra model calls, reuse OF3 outputs
                of_structure = of_metrics.get("structure_path")
                plddt = of_metrics.get("plddt_per_atom")
                if of_structure:
                    from binding_metrics.metrics.evobind import (
                        compute_evobind_adversarial_check,
                        compute_evobind_score,
                    )

                    # Primary score on the OF3 prediction
                    try:
                        evobind = compute_evobind_score(
                            of_structure,
                            plddt_per_atom=plddt,
                            binder_chain=peptide_chain,
                            receptor_chain=receptor_chain,
                        )
                        _merge_reason(of_metrics, evobind, "evobind")
                        of_metrics.update(evobind)
                    except Exception as e:
                        _warn(f"EvoBind score failed: {e}")
                        of_metrics["evobind_error"] = str(e)

                    # Adversarial check: does the OF3 prediction agree with the
                    # input design pose? Large ΔCOM = OF3 places the binder
                    # elsewhere → design pose not supported by the prediction.
                    try:
                        adversarial = compute_evobind_adversarial_check(
                            design_structure_path=input_path,
                            afm_structure_path=of_structure,
                            binder_chain=peptide_chain,
                            receptor_chain=receptor_chain,
                            afm_plddt_per_atom=plddt,
                        )
                        _merge_reason(of_metrics, adversarial, "evobind adversarial")
                        of_metrics.update(adversarial)
                    except Exception as e:
                        _warn(f"EvoBind adversarial check failed: {e}")
                        of_metrics["adversarial_error"] = str(e)

                results["openfold"] = of_metrics

        except Exception as e:
            _warn(f"OpenFold failed: {e}")
            traceback.print_exc()
            results["openfold"] = {"error": str(e)}
    else:
        results["openfold"] = {"skipped": True}

    return results


def _collect_failures(results: dict) -> list:
    """Return ``(step, reason)`` for every metric step that did not complete.

    A step counts as failed when its result dict carries an ``error`` key, or
    when a step that reports a ``success`` flag (relaxation, energy) has
    ``success is False``. Steps marked ``{"skipped": True}`` are not failures.
    Used to make the pipeline exit non-zero instead of silently reporting
    success when metrics could not be computed.
    """
    failures = []
    for step, res in results.items():
        if not isinstance(res, dict) or res.get("skipped"):
            continue
        if "error" in res:
            failures.append((step, str(res["error"])[:200]))
        elif res.get("success") is False:
            reason = res.get("error_message") or "step reported success=False"
            failures.append((step, str(reason)[:200]))
    return failures


def _parse_metrics(value: str) -> frozenset:
    """Parse ``--metrics``: a comma-separated list of names from ``KNOWN_METRICS``."""
    names = {v.strip() for v in value.split(",")}
    unknown = names - KNOWN_METRICS
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown metric(s): {', '.join(sorted(unknown))}. "
            f"Valid choices: {', '.join(sorted(KNOWN_METRICS))}"
        )
    return frozenset(names)


def main():
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Run the full binding-metrics pipeline on a single structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input CIF or PDB file")
    parser.add_argument(
        "--output-dir", "-o", type=Path, required=True, help="Directory to write all outputs"
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Sample identifier (defaults to input file stem)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=DEFAULT_DEVICE,
        help=f"Compute device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--peptide-chain",
        "--binder-chain",
        action=ChainAliasAction,
        type=str,
        default=None,
        help="Peptide chain ID (auto-detect if omitted)",
    )
    parser.add_argument(
        "--receptor-chain",
        "--target-chain",
        action=ChainAliasAction,
        type=str,
        default=None,
        help="Receptor chain ID (auto-detect if omitted)",
    )
    parser.add_argument(
        "--reference",
        "--native",
        dest="reference",
        type=Path,
        default=None,
        help="Reference/native complex for reference-based accuracy "
        "metrics (DockQ, fnat, i-RMSD, L-RMSD). Supplying this "
        "auto-enables the 'dockq' metric. Requires: pip install DockQ",
    )

    # Prep
    prep_group = parser.add_argument_group("Preparation")
    prep_group.add_argument(
        "--skip-prep", action="store_true", help="Skip PDBFixer prep; run relax on raw input"
    )
    prep_group.add_argument(
        "--ph",
        type=float,
        default=DEFAULT_PH,
        help=f"pH for hydrogen placement during prep (default: {DEFAULT_PH})",
    )
    prep_group.add_argument(
        "--keep-water",
        action="store_true",
        help="Retain crystallographic water molecules during prep",
    )
    prep_group.add_argument(
        "--canonicalize",
        action="store_true",
        help=(
            "Replace non-standard residues with standard equivalents during prep "
            "(e.g. MSE→MET, SEP→SER). By default they are preserved for GAFF2 "
            "parameterisation in relax (--small-molecules auto)."
        ),
    )

    # Relaxation
    relax_group = parser.add_argument_group("Relaxation")
    relax_group.add_argument(
        "--skip-relax", action="store_true", help="Skip relaxation; run metrics on raw input"
    )
    relax_group.add_argument(
        "--md-duration-ps",
        type=float,
        default=DEFAULT_MD_DURATION_PS,
        help=f"MD duration in ps (0 = minimize only, default: {DEFAULT_MD_DURATION_PS:g})",
    )
    relax_group.add_argument(
        "--random-seed",
        type=_seed_arg,
        default=DEFAULT_RANDOM_SEED,
        metavar="INT|none",
        help=(
            "Seed for all stochastic steps (hydrogen placement, MD velocities "
            "and thermostat). A fixed integer makes the run reproducible "
            f"(default: {DEFAULT_RANDOM_SEED}); pass 'none' for fresh randomness "
            "each run, e.g. to generate independent MD replicas."
        ),
    )

    # Metrics
    metrics_group = parser.add_argument_group("Metrics")
    metrics_group.add_argument(
        "--metrics",
        type=_parse_metrics,
        default=ALL_METRICS,
        metavar="METRICS",
        help=(
            "Comma-separated list of metrics to compute. "
            f"Valid: {', '.join(sorted(KNOWN_METRICS))}. "
            "Default: all reference-free metrics. 'dockq' also needs --reference."
        ),
    )
    metrics_group.add_argument(
        "--energy-modes",
        nargs="+",
        choices=["raw", "relaxed", "after_md"],
        default=["relaxed"],
        help="Energy evaluation modes (default: relaxed)",
    )

    # OpenFold
    openfold_group = parser.add_argument_group("OpenFold")
    openfold_group.add_argument(
        "--openfold-mode",
        choices=["score", "refold"],
        default="score",
        help="score: both chains as templates (confidence); "
        "refold: binder predicted freely (refolding RMSD). "
        "Default: score",
    )
    openfold_group.add_argument(
        "--openfold-conda-env",
        type=str,
        default="openfold3",
        help="Conda environment name where OpenFold3 is installed "
        "(default: openfold3). Set to empty string to use "
        "the current environment if openfold3 is installed there.",
    )
    add_openfold_seeds_arg(openfold_group)

    # Report
    report_group = parser.add_argument_group("Report")
    report_group.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        dest="fmt",
        help="Results output format (default: json)",
    )
    report_group.add_argument(
        "--summary",
        action="store_true",
        help="Also write a human-readable summary (*_report.md or *_report.html)",
    )
    report_group.add_argument(
        "--summary-format",
        choices=["md", "html"],
        default="md",
        dest="summary_format",
        help="Summary format (default: md)",
    )
    from binding_metrics.cli import add_log_file_arg

    add_log_file_arg(report_group)
    add_config_arg(parser)

    args = parse_args_with_config(parser)

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # A reference auto-enables DockQ; validate it exists up front.
    metrics = args.metrics
    if args.reference is not None:
        if not args.reference.exists():
            print(f"ERROR: reference file not found: {args.reference}", file=sys.stderr)
            sys.exit(1)
        metrics = metrics | {"dockq"}

    from binding_metrics.cli import log_to_file

    with log_to_file(args.log_file):
        sample_id = args.sample_id or args.input.stem
        print(f"\n{'#' * 60}")
        print(f"  binding-metrics-run: {sample_id}")
        print(f"  Input:  {args.input}")
        print(f"  Output: {args.output_dir}")
        if args.log_file:
            print(f"  Log:    {args.log_file}")
        print(f"{'#' * 60}")

        t_total = time.time()
        try:
            results = run_pipeline(
                input_path=args.input,
                output_dir=args.output_dir,
                sample_id=sample_id,
                skip_prep=args.skip_prep,
                ph=args.ph,
                keep_water=args.keep_water,
                canonicalize=args.canonicalize,
                skip_relax=args.skip_relax,
                md_duration_ps=args.md_duration_ps,
                device=args.device,
                peptide_chain=args.peptide_chain,
                receptor_chain=args.receptor_chain,
                metrics=metrics,
                energy_modes=tuple(args.energy_modes),
                reference_path=args.reference,
                openfold_mode=args.openfold_mode,
                openfold_conda_env=args.openfold_conda_env,
                random_seed=args.random_seed,
                openfold_seeds=args.openfold_seeds,
            )
        except ChainNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        results["total_elapsed_s"] = round(time.time() - t_total, 1)

        from binding_metrics.protocols.report import write_report

        results_path = write_report(
            results,
            args.output_dir,
            sample_id,
            fmt=args.fmt,
            summary=args.summary,
            summary_format=args.summary_format,
        )

        failures = _collect_failures(results)
        if failures:
            print(f"\n{'#' * 60}")
            print(
                f"  FAILED in {results['total_elapsed_s']}s — "
                f"{len(failures)} step(s) did not complete:"
            )
            for step, reason in failures:
                print(f"    [x] {step}: {reason}")
            print(f"  Partial results: {results_path}")
            print(f"{'#' * 60}\n")
            sys.exit(1)

        print(f"\n{'#' * 60}")
        print(f"  DONE in {results['total_elapsed_s']}s")
        print(f"  Results: {results_path}")
        print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
