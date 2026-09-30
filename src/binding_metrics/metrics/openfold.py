"""OpenFold3 structure prediction metrics.

Provides utilities to run OpenFold3 inference and extract confidence metrics
from its output files.

OpenFold3 is a fully open-source (Apache 2.0) biomolecular structure prediction
model based on AlphaFold3, developed by the OpenFold Consortium. It predicts
structures of proteins, RNA, DNA, and small-molecule complexes.

Output files per prediction (seed S, sample M):
  {output_dir}/{query_name}/seed_{S}/{prefix}_confidences_aggregated.json
      Scalar confidence metrics: avg_plddt, gpde, ptm, iptm, chain_ptm,
      chain_pair_iptm, bespoke_iptm, disorder, has_clash, sample_ranking_score
  {output_dir}/{query_name}/seed_{S}/{prefix}_confidences.json (or .npz)
      Per-atom arrays: plddt[n_atoms], pde[n_tokens, n_tokens],
      pae[n_tokens, n_tokens]; written unless the run set write_full_confidence_scores false
  {output_dir}/{query_name}/seed_{S}/{prefix}_model.cif (or .cif.gz, .pdb)
      3D structure (pLDDT in B-factor column)
  {output_dir}/{query_name}/seed_{S}/timing.json
      Runtime (excluding MSA computation)

Layout: this module parses outputs (``compute_openfold_metrics``) and runs
inference (``run_openfold`` and the ``run_openfold_*`` wrappers). The runner
YAML and query preparation live in ``_openfold_run.py`` and the command line in
``_openfold_cli.py``; every name they define is re-exported here.

References:
  The OpenFold3 Team (2026) OpenFold3, v0.5.0 (OpenBind-0 weights). Software,
  doi:10.5281/zenodo.22042719, github.com/aqlaboratory/openfold-3
  Abramson et al. (2024) Accurate structure prediction of biomolecular
  interactions with AlphaFold 3. Nature 630:493-500. Defines the outputs parsed
  here (pLDDT, PAE, PDE, pTM, ipTM) and the tokenisation that the interface
  PDE/PAE slicing relies on: one token per standard residue, one per heavy atom
  for ligands and modified residues.

Usage (Python API):
    from binding_metrics import compute_openfold_metrics

    metrics = compute_openfold_metrics(
        output_dir="./openfold_out",
        query_name="my_complex",
        seed=1,
        sample=1,
    )
    print(metrics["avg_plddt"], metrics["ptm"], metrics["iptm"])

Usage (CLI):
    # Parse existing output:
    binding-metrics-openfold parse --output-dir ./openfold_out --query-name my_complex

    # Run inference then parse:
    binding-metrics-openfold run \\
        --query-json query.json \\
        --output-dir ./openfold_out \\
        --query-name my_complex
"""

import subprocess  # noqa: F401  (tests patch openfold.subprocess)
from pathlib import Path
from typing import Optional, Sequence

from binding_metrics.metrics._common import resolve_chain_role
from binding_metrics.metrics._openfold_cli import (  # noqa: F401  (re-exported)
    _add_parse_args,
    _add_query_seeds_arg,
    _print_metrics,
    main,
)
from binding_metrics.metrics._openfold_run import (  # noqa: F401  (re-exported)
    _DEFAULT_MODEL_PRESETS,
    _DEFAULT_QUERY_SEEDS,
    OpenFoldQueryError,
    OpenFoldRunError,
    UnmappableResidueError,
    _BatchSample,
    _extract_chain_to_cif,
    _extract_sequence_from_structure,
    _query_seeds,
    _run_openfold_command,
    _safe_entry_id,
    _write_a3m_self_alignment,
    _write_runner_yaml,
    prepare_batched_refolding_queries,
    prepare_batched_scoring_queries,
    prepare_refolding_query,
    prepare_scoring_query,
)
from binding_metrics.metrics.prediction import summarize_prediction
from binding_metrics.predictors._confidence import (  # noqa: F401  (re-exported)
    _binder_ca_rmsd,
    _binder_plddt_per_residue,
    _chain_token_offsets,
    _check_token_offsets,
    _import_biotite_struc,
    _interface_pae_stats,
    _interface_pde_stats,
    _load_atoms,
)
from binding_metrics.predictors.of3 import (  # noqa: F401  (re-exported)
    OpenFold3Parser,
)
from binding_metrics.predictors.of3 import (  # noqa: F401  (re-exported)
    parse_aggregated_confidences as _parse_confidences_aggregated,
)
from binding_metrics.predictors.of3 import parse_full_confidences as _parse_confidences
from binding_metrics.predictors.of3 import (  # noqa: F401  (re-exported)
    parse_timing as _parse_timing,
)
from binding_metrics.predictors.registry import get_parser

# ---------------------------------------------------------------------------
# Output file discovery and parsers (the OpenFold3 adapter of binding_metrics.predictors)
# ---------------------------------------------------------------------------


def _find_prediction_files(
    output_dir: Path,
    query_name: str,
    seed: int = 1,
    sample: int = 1,
) -> dict[str, Optional[Path]]:
    """Locate OpenFold3 output files for a given seed/sample.

    Args:
        output_dir: Top-level OpenFold3 output directory.
        query_name: Query name as specified in the input JSON.
        seed: Seed index (1-based position of the ``seed_*`` directory in the numeric order
            of the seed values), not a seed value. OF3 transforms input seeds, so this
            selects by position rather than value.
        sample: Sample index (default 1).

    Returns:
        Dict with keys: ``structure``, ``confidences``, ``confidences_aggregated``,
        ``timing``. Values are resolved Paths or None if not found. The structure may be a
        ``.cif``, ``.cif.gz`` or ``.pdb`` file.
    """
    files = OpenFold3Parser().find_files(
        Path(output_dir), query_name, seed_index=seed, sample=sample
    )
    return {
        "structure": files.structure,
        "confidences": files.arrays,
        "confidences_aggregated": files.scores,
        "timing": files.timing,
    }


# ---------------------------------------------------------------------------
# Interface PAE from a confidences file
# ---------------------------------------------------------------------------
# The slicing, per-residue pLDDT and RMSD helpers are model-agnostic and live in
# binding_metrics.predictors._confidence; their private names are imported above.


def compute_interface_pae(
    confidences_path,
    structure_path,
    binder_chain: str,
    receptor_chain: Optional[str] = None,
    *,
    target_chain: Optional[str] = None,
) -> dict:
    """Interface PAE slice from OpenFold3 output.

    OpenFold3 (>= v0.4.1) always computes the PAE head and writes the full
    ``pae`` matrix to ``*_confidences.json/.npz`` alongside ``plddt`` and
    ``pde``, unless the run set ``write_full_confidence_scores`` to false. This
    loads that matrix and the predicted structure and returns the interface
    (binder×receptor) PAE statistics.

    Args:
        confidences_path: Path to ``*_confidences.json`` or ``.npz``.
        structure_path: Path to the predicted model (.cif/.pdb) — used to map
            chains to PAE token ranges.
        binder_chain: Chain ID of the binder.
        receptor_chain: Chain ID of the receptor. Required, through this
            parameter or ``target_chain``.
        target_chain: Alias of ``receptor_chain``; different IDs in both raise
            ``ValueError``.

    Returns:
        Dict from :func:`_interface_pae_stats` (``pae_interface``,
        ``mean_interface_pae``, ``max_interface_pae``, token counts).

    Raises:
        ValueError: If the confidences file has no PAE matrix (it comes from a
            run with ``write_full_confidence_scores`` false, or from a version
            before 0.4), or if the matrix size does not equal the structure's
            residue count (a ligand or modified residue makes the token count
            differ).
    """
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain, required=True
    )
    conf = _parse_confidences(Path(confidences_path))
    pae = conf.get("pae")
    if pae is None:
        raise ValueError(
            f"No PAE matrix in {confidences_path}. OpenFold3 0.4 and later write the 'pae' "
            "array to the full confidences file unless write_full_confidence_scores is "
            "false, so this file comes from such a run or from an older version."
        )
    atoms = _load_atoms(Path(structure_path))
    return _interface_pae_stats(pae, atoms, binder_chain, receptor_chain)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def compute_openfold_metrics(
    output_dir: str | Path,
    query_name: str,
    seed: int = 1,
    sample: int = 1,
    include_matrices: bool = False,
    reference_structure_path: Optional[str | Path] = None,
    binder_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    *,
    seed_index: Optional[int] = None,
    target_chain: Optional[str] = None,
) -> dict:
    """Extract confidence metrics from OpenFold3 output files.

    **Mode 1 — confidence metrics only (no reference):**
    Parse the aggregated and per-atom confidence files to get scalar metrics
    (avg_plddt, ptm, iptm, chain scores) and optionally per-residue binder
    pLDDT and interface PDE statistics. Pass ``binder_chain`` (and optionally
    ``receptor_chain``) to enable per-chain analysis.

    **Mode 1 with reference — refolding RMSD:**
    Pass ``reference_structure_path`` together with ``binder_chain`` (and
    ``receptor_chain`` for receptor-frame alignment) to additionally compute
    the binder Cα RMSD between the OF3 prediction and a known reference
    structure (e.g., crystal or MD-relaxed). This measures how faithfully OF3
    recovers the bound binder conformation.

    Args:
        output_dir: Top-level OpenFold3 output directory.
        query_name: Query name as specified in the input JSON (used to locate
            the ``{output_dir}/{query_name}/`` subdirectory).
        seed: 1-based index into the sorted ``seed_*`` directories of the query,
            not a random seed value: OpenFold3 names the directories after its
            own transformed seeds, so they are selected by position (default 1).
        sample: Sample index to parse (default 1).
        include_matrices: If True, include the full PDE matrix in the result
            (can be large). Default False.
        reference_structure_path: Optional path to a reference CIF/PDB
            (e.g., crystal structure). When supplied together with
            ``binder_chain``, the binder Cα RMSD between the OF3 prediction
            and this reference is computed and stored as ``binder_ca_rmsd``.
        binder_chain: Chain ID of the binder/ligand in the predicted structure.
            Enables per-residue pLDDT for the binder, interface PDE stats
            (when ``receptor_chain`` is also given), and binder RMSD
            (when ``reference_structure_path`` is also given).
        receptor_chain: Chain ID of the receptor/target. When provided
            alongside ``binder_chain``:
              - Interface PDE statistics are computed (mean/max PDE for the
                binder×receptor token block).
              - Receptor Cα atoms are used as the superposition reference
                when computing ``binder_ca_rmsd``, giving the
                receptor-frame binder RMSD.
        seed_index: Clearer name for ``seed``; when given it takes precedence.
        target_chain: Alias of ``receptor_chain``; different IDs in both raise
            ``ValueError``.

    Returns:
        Dictionary with keys:

        Structure:
            structure_path (str | None): path to the predicted .cif/.pdb file
            query_name (str), seed (int, the seed index used), sample (int)

        Scalar confidence metrics [from confidences_aggregated.json]:
            avg_plddt (float): mean pLDDT across all atoms [0–100]
            gpde (float): global predicted distance error (Å)
            ptm (float): predicted TM-score [0–1]; NaN if the aggregated file lacks it
            iptm (float): interface pTM [0–1]; NaN if single chain or the aggregated file lacks it
            disorder (float): average relative SASA [0–1]
            has_clash (float): 1.0 if steric clashes detected, 0.0 otherwise
            sample_ranking_score (float): weighted composite score for ranking
            chain_ptm (dict): per-chain pTM scores {chain_id: float}
            chain_pair_iptm (dict): pairwise interface pTM, keyed by the strings
                OpenFold3 writes: {"(A, B)": float}
            bespoke_iptm (dict): bespoke interface score, keyed the same way

        Per-atom / per-token data [from confidences.json]:
            plddt_per_atom (np.ndarray | None): per-atom pLDDT, shape (n_atoms,)
            n_atoms (int): number of atoms
            pde (np.ndarray | None): PDE matrix (n_tokens×n_tokens); only if
                include_matrices=True
            max_pde (float): max PDE value (Å)
            pae (np.ndarray | None): PAE matrix (n_tokens×n_tokens); only if
                include_matrices=True and the full confidence file has a PAE matrix
            max_pae (float): max PAE value (Å); NaN if no PAE matrix present

        Per-chain structural analysis [requires binder_chain]:
            binder_plddt_per_residue (np.ndarray | None): mean pLDDT per
                residue for the binder chain, shape (n_binder_res,)
            binder_avg_plddt (float): mean pLDDT over all binder residues

        Interface PDE / PAE [requires binder_chain + receptor_chain]:
            The block is located with one token per residue. When the matrix
            size differs from the structure's residue count (a ligand, ion or
            modified residue is tokenised per atom), the interface values stay
            NaN, a warning is issued and ``reason`` says why.
            mean_interface_pde (float): mean PDE over binder×receptor tokens (Å)
            max_interface_pde (float): max PDE over binder×receptor tokens (Å)
            pde_interface (np.ndarray | None): raw PDE slice, shape
                (n_binder_res, n_receptor_res); only if include_matrices=True
            mean_interface_pae (float): mean PAE over the interface tokens (Å),
                averaged over both slice directions; NaN if there is no PAE matrix
            max_interface_pae (float): max PAE over the interface tokens (Å)
            pae_interface (np.ndarray | None): raw PAE slice (binder→receptor);
                only if include_matrices=True

        Refolding RMSD [requires binder_chain + reference_structure_path]:
            binder_ca_rmsd (float): binder Cα RMSD vs. reference (Å).
                Computed in the receptor frame if receptor_chain is given
                (predicted structure superposed on receptor Cα first).

        Timing:
            timing (dict): runtime entries from timing.json, empty if absent

        Failures:
            reason (str): present only when a value could not be computed (it
                keeps its NaN or None sentinel). Names each affected analysis
                and its cause, separated by "; ": missing output files, a
                missing model structure, a pLDDT or PDE/PAE array that does not
                fit the structure, a binder RMSD mismatch, or a structure that
                could not be parsed.
    """
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain
    )
    if seed_index is not None:
        seed = seed_index
    record = get_parser("of3").load(output_dir, query_name, seed_index=seed, sample=sample)
    result = summarize_prediction(
        record,
        include_matrices=include_matrices,
        reference_structure_path=reference_structure_path,
        binder_chain=binder_chain,
        receptor_chain=receptor_chain,
        caller="compute_openfold_metrics",
        stacklevel=3,  # the caller of compute_openfold_metrics, through summarize_prediction
    )
    del result["model"]  # this function has always returned the OpenFold3 keys only
    return result


# ---------------------------------------------------------------------------
# Runner: invoke OpenFold3 as a subprocess
# ---------------------------------------------------------------------------


def run_openfold(
    query_json: str | Path,
    output_dir: str | Path,
    inference_ckpt_path: Optional[str | Path] = None,
    num_diffusion_samples: int = 5,
    num_model_seeds: int = 1,
    use_msa_server: bool = True,
    model_presets: Optional[list[str]] = None,
    runner_yaml: Optional[str | Path] = None,
    extra_args: Optional[list[str]] = None,
    conda_env: Optional[str] = None,
    template_dir: Optional[str | Path] = None,
) -> Path:
    """Run OpenFold3 inference as a subprocess.

    Invokes ``run_openfold predict`` from the OpenFold3 package.
    OpenFold3 must be installed in either the current environment or a named
    conda environment (see ``conda_env``).

    Args:
        query_json: Path to the input JSON file describing the prediction query.
        output_dir: Directory where OpenFold3 writes predictions.
        inference_ckpt_path: Optional path to a model checkpoint (.pt file).
            Uses the default downloaded checkpoint if None.
        num_diffusion_samples: Number of structure samples per query (default 5).
        num_model_seeds: Passed to OpenFold3 as ``--num_model_seeds``. The seed
            values of a query are the ``"seeds"`` list in ``query_json``, which
            the ``prepare_*`` functions set from their ``seeds`` argument
            (default ``[42]``).
        use_msa_server: Use the ColabFold MSA server for alignment generation
            (default True). MSAs then come from a remote service, so results can
            change over time, and the sequences leave the machine. Set False if
            MSAs are pre-computed.
        model_presets: List of model configuration presets. These are written to
            a runner YAML and passed via ``--runner_yaml``. The ``"predict"``
            preset is always prepended if not already present. Available presets:

            - ``"predict"`` — required base preset for inference
            - ``"low_mem"`` — memory-efficient mode; pairformer embeddings are
              computed sequentially. Recommended for large complexes or limited
              GPU memory. Significant slowdown with many diffusion samples.

            Defaults to ``["predict", "low_mem"]``. OpenFold3 0.4.1 removed the
            ``"pae_enabled"`` preset: the PAE head is on by default and pTM, ipTM
            and PAE are always written. A ``"pae_enabled"`` entry in the list is
            left out of the runner YAML with a ``DeprecationWarning``.
            Ignored if ``runner_yaml`` is also provided.
        runner_yaml: Explicit path to a runner YAML configuration file. Overrides
            ``model_presets`` when both are provided. CLI flags always take
            precedence over YAML values.
        extra_args: Additional CLI arguments passed verbatim.
        conda_env: Name of the conda environment where OpenFold3 is installed
            (e.g. ``"openfold3"``). When given, the command is wrapped as
            ``conda run -n {conda_env} --no-capture-output run_openfold ...``.
            If None, ``run_openfold`` must be on the current PATH.

    Returns:
        Path to the output directory.

    Raises:
        FileNotFoundError: If ``run_openfold`` is not on PATH and no
            ``conda_env`` is specified.
        OpenFoldRunError: If OpenFold3 exits non-zero. It is a
            ``subprocess.CalledProcessError``; its message starts with the failing line of
            stderr and adds a fix for missing or incompatible weights, GPU memory and
            ``/dev/shm``.
        OpenFoldQueryError: If OpenFold3 exits with status 0 but every query failed, which
            it reports only in ``summary.txt`` and ``logs/``. A partial failure is logged.
    """
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ``--openfold-conda-env ""`` means the current environment, as its help says.
    # run_openfold_scoring, run_openfold_refolding and run_openfold_batched end here.
    conda_env = conda_env or None
    if conda_env is None and shutil.which("run_openfold") is None:
        raise FileNotFoundError(
            "run_openfold not found on PATH. "
            "Pass conda_env='openfold3' (or whichever env has OF3 installed)."
        )

    # Resolve runner YAML: explicit path takes precedence over model_presets
    effective_yaml: Optional[Path] = None
    if runner_yaml is not None:
        # TODO(#96): template_dir is dropped here, so a given runner_yaml loses the templates.
        effective_yaml = Path(runner_yaml)
    else:
        presets = list(model_presets) if model_presets is not None else list(_DEFAULT_MODEL_PRESETS)
        if "predict" not in presets:
            presets.insert(0, "predict")
        effective_yaml = _write_runner_yaml(
            output_dir,
            presets,
            template_dir=Path(template_dir) if template_dir is not None else None,
            conda_env=conda_env,
        )

    of3_cmd = [
        "run_openfold",
        "predict",
        f"--query_json={query_json}",
        f"--output_dir={output_dir}",
        f"--num_diffusion_samples={num_diffusion_samples}",
        f"--num_model_seeds={num_model_seeds}",
        f"--use_msa_server={str(use_msa_server).lower()}",
        f"--runner_yaml={effective_yaml}",
    ]

    if inference_ckpt_path is not None:
        of3_cmd.append(f"--inference_ckpt_path={inference_ckpt_path}")

    if extra_args:
        of3_cmd.extend(extra_args)

    if conda_env is not None:
        cmd = ["conda", "run", "-n", conda_env, "--no-capture-output"] + of3_cmd
    else:
        cmd = of3_cmd

    _run_openfold_command(cmd, output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Wrappers: prepare the query, then run OpenFold3 (scoring and refolding)
# ---------------------------------------------------------------------------


def run_openfold_scoring(
    complex_structure_path: str | Path,
    receptor_chain: str,
    binder_chain: str,
    query_name: str,
    output_dir: str | Path,
    template_cif_path: Optional[str | Path] = None,
    inference_ckpt_path: Optional[str | Path] = None,
    num_diffusion_samples: int = 5,
    num_model_seeds: int = 1,
    use_msa_server: bool = True,
    model_presets: Optional[list[str]] = None,
    runner_yaml: Optional[str | Path] = None,
    extra_args: Optional[list[str]] = None,
    conda_env: Optional[str] = None,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
    *,
    on_unmappable_residue: str = "error",
) -> Path:
    """Run OpenFold3 scoring of an existing complex structure (Mode 1).

    Both receptor and binder chains are provided as structural templates.
    OF3 scores the known conformation and outputs confidence metrics.
    Use :func:`compute_openfold_metrics` with ``binder_chain`` and
    ``receptor_chain`` to extract per-chain scores after inference.

    Args:
        complex_structure_path: CIF/PDB of the full complex.
        receptor_chain: Chain ID of the receptor.
        binder_chain: Chain ID of the binder.
        query_name: Prediction query name.
        output_dir: Top-level output directory.
        template_cif_path: Optional pre-prepared complex CIF (e.g., MD-relaxed).
        inference_ckpt_path: Optional model checkpoint path.
        num_diffusion_samples: Structure samples per query (default 5).
        num_model_seeds: Passed to OpenFold3 as ``--num_model_seeds``. The seed
            values themselves are set by ``seeds``.
        use_msa_server: Use the ColabFold MSA server (default True). MSAs then
            come from a remote service, so results can change over time, and
            the sequences leave the machine. Pass False with pre-computed MSAs.
        model_presets: Model configuration presets.
        runner_yaml: Explicit runner YAML; overrides ``model_presets``.
        extra_args: Additional CLI args for OF3.
        conda_env: Conda environment where OpenFold3 is installed.
        seeds: Seed values written to the query JSON (default ``(42,)``).
        on_unmappable_residue: ``"error"`` (default) raises
            :class:`UnmappableResidueError` before anything is written or started
            when a residue has no one-letter code or CCD code OpenFold3 can take;
            ``"x"`` sends an ``X`` for it and logs a warning.

    Returns:
        Path to the OF3 predictions output directory
        (``{output_dir}/predictions/``).

    Raises:
        UnmappableResidueError: See ``on_unmappable_residue``; raised before any
            file is written or process started.
    """
    output_dir = Path(output_dir)
    query_dir = output_dir / "query"
    predictions_dir = output_dir / "predictions"

    query_json = prepare_scoring_query(
        complex_structure_path=complex_structure_path,
        receptor_chain=receptor_chain,
        binder_chain=binder_chain,
        query_name=query_name,
        output_dir=query_dir,
        template_cif_path=template_cif_path,
        seeds=seeds,
        on_unmappable_residue=on_unmappable_residue,
    )

    run_openfold(
        query_json=query_json,
        output_dir=predictions_dir,
        inference_ckpt_path=inference_ckpt_path,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        use_msa_server=use_msa_server,
        model_presets=model_presets,
        runner_yaml=runner_yaml,
        extra_args=extra_args,
        conda_env=conda_env,
        template_dir=query_dir / "templates",
    )
    return predictions_dir


def run_openfold_refolding(
    complex_structure_path: str | Path,
    receptor_chain: str,
    binder_chain: str,
    query_name: str,
    output_dir: str | Path,
    template_cif_path: Optional[str | Path] = None,
    inference_ckpt_path: Optional[str | Path] = None,
    num_diffusion_samples: int = 5,
    num_model_seeds: int = 1,
    use_msa_server: bool = True,
    model_presets: Optional[list[str]] = None,
    runner_yaml: Optional[str | Path] = None,
    extra_args: Optional[list[str]] = None,
    conda_env: Optional[str] = None,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
    *,
    on_unmappable_residue: str = "error",
) -> Path:
    """Run OpenFold3 refolding: binder predicted freely, receptor fixed as template.

    Convenience wrapper for Mode 2. Calls :func:`prepare_refolding_query` to
    build the input JSON, then invokes :func:`run_openfold`. After inference,
    call :func:`compute_openfold_metrics` with ``binder_chain``,
    ``receptor_chain``, and ``reference_structure_path`` to get refolding
    RMSD and per-chain confidence metrics.

    Directory layout::

        {output_dir}/
          query/            — query JSON, A3M, and template CIF
          predictions/      — OF3 output (structures, confidence files)

    Args:
        complex_structure_path: CIF/PDB of the full complex.
        receptor_chain: Chain ID of the receptor (fixed as template).
        binder_chain: Chain ID of the binder (refolded from sequence only).
        query_name: Prediction query name.
        output_dir: Top-level output directory.
        template_cif_path: Optional pre-prepared receptor template CIF.
        inference_ckpt_path: Optional model checkpoint path.
        num_diffusion_samples: Structure samples per query (default 5).
        num_model_seeds: Passed to OpenFold3 as ``--num_model_seeds``. The seed
            values themselves are set by ``seeds``.
        use_msa_server: Use the ColabFold MSA server (default True). MSAs then
            come from a remote service, so results can change over time, and
            the sequences leave the machine. Pass False with pre-computed MSAs.
        model_presets: Model configuration presets.
        runner_yaml: Explicit runner YAML; overrides ``model_presets``.
        extra_args: Additional CLI args for OF3, passed verbatim. The receptor
            template reaches OpenFold3 through ``template_preprocessor_settings.
            structure_directory`` of the runner YAML, which points to
            ``{output_dir}/query/templates/``; there is no command-line flag for
            it.
        conda_env: Conda environment where OpenFold3 is installed.
        seeds: Seed values written to the query JSON (default ``(42,)``).
        on_unmappable_residue: ``"error"`` (default) raises
            :class:`UnmappableResidueError` before anything is written or started
            when a residue has no one-letter code or CCD code OpenFold3 can take;
            ``"x"`` sends an ``X`` for it and logs a warning.

    Returns:
        Path to the OF3 predictions output directory
        (``{output_dir}/predictions/``).

    Raises:
        UnmappableResidueError: See ``on_unmappable_residue``; raised before any
            file is written or process started.
    """
    output_dir = Path(output_dir)
    query_dir = output_dir / "query"
    predictions_dir = output_dir / "predictions"

    query_json = prepare_refolding_query(
        complex_structure_path=complex_structure_path,
        receptor_chain=receptor_chain,
        binder_chain=binder_chain,
        query_name=query_name,
        output_dir=query_dir,
        template_cif_path=template_cif_path,
        seeds=seeds,
        on_unmappable_residue=on_unmappable_residue,
    )

    run_openfold(
        query_json=query_json,
        output_dir=predictions_dir,
        inference_ckpt_path=inference_ckpt_path,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        use_msa_server=use_msa_server,
        model_presets=model_presets,
        runner_yaml=runner_yaml,
        extra_args=extra_args,
        conda_env=conda_env,
        template_dir=query_dir / "templates",
    )
    return predictions_dir


# ---------------------------------------------------------------------------
# Batched inference (multiple queries in one OF3 subprocess)
# ---------------------------------------------------------------------------


def run_openfold_batched(
    samples: list[_BatchSample],
    output_dir: str | Path,
    mode: str = "score",
    inference_ckpt_path: Optional[str | Path] = None,
    num_diffusion_samples: int = 5,
    num_model_seeds: int = 1,
    use_msa_server: bool = True,
    model_presets: Optional[list[str]] = None,
    runner_yaml: Optional[str | Path] = None,
    extra_args: Optional[list[str]] = None,
    conda_env: Optional[str] = None,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
    *,
    on_unmappable_residue: str = "error",
) -> Path:
    """Run OpenFold3 inference on multiple samples in a single subprocess.

    Prepares all queries into one combined JSON, calls ``run_openfold``
    once (amortising model loading and CUDA initialisation), and returns
    the predictions directory.  Use :func:`compute_openfold_metrics` with
    each sample's ``query_name`` to extract per-sample results.

    Args:
        samples: Per-sample descriptors.
        output_dir: Top-level output directory.
        mode: ``"score"`` (both chains as templates) or ``"refold"``
            (binder predicted from sequence only).
        inference_ckpt_path: Optional model checkpoint path.
        num_diffusion_samples: Structure samples per query (default 5).
        num_model_seeds: Passed to OpenFold3 as ``--num_model_seeds``. The seed
            values themselves are set by ``seeds``.
        use_msa_server: Use the ColabFold MSA server (default True). MSAs then
            come from a remote service, so results can change over time, and
            the sequences leave the machine. Pass False with pre-computed MSAs.
        model_presets: Model configuration presets.
        runner_yaml: Explicit runner YAML; overrides ``model_presets``.
        extra_args: Additional CLI args for OF3.
        conda_env: Conda environment name (default None).
        seeds: Seed values written to the query JSON (default ``(42,)``).
        on_unmappable_residue: ``"error"`` (default) raises
            :class:`UnmappableResidueError` before anything is written or started
            when a residue has no one-letter code or CCD code OpenFold3 can take;
            ``"x"`` sends an ``X`` for it and logs a warning.

    Returns:
        Path to the OF3 predictions output directory.

    Raises:
        UnmappableResidueError: See ``on_unmappable_residue``; the error names every
            affected sample and is raised before any file is written or process started.
    """
    output_dir = Path(output_dir)
    query_dir = output_dir / "query"
    predictions_dir = output_dir / "predictions"

    prepare = (
        prepare_batched_refolding_queries if mode == "refold" else prepare_batched_scoring_queries
    )
    query_json = prepare(
        samples, query_dir, seeds=seeds, on_unmappable_residue=on_unmappable_residue
    )

    run_openfold(
        query_json=query_json,
        output_dir=predictions_dir,
        inference_ckpt_path=inference_ckpt_path,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        use_msa_server=use_msa_server,
        model_presets=model_presets,
        runner_yaml=runner_yaml,
        extra_args=extra_args,
        conda_env=conda_env,
        template_dir=query_dir / "templates",
    )
    return predictions_dir


if __name__ == "__main__":
    main()
