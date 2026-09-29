"""OpenFold3 runner configuration and query preparation.

Builds the runner YAML and the query JSON (with template CIFs and A3M
self-alignments) that ``run_openfold predict`` reads. The subprocess call and the
``run_openfold_*`` wrappers stay in :mod:`binding_metrics.metrics.openfold`, which
re-exports everything defined here.
"""

import dataclasses
import json
from pathlib import Path
from typing import Optional, Sequence

#: Seed values written into the ``"seeds"`` field of the query JSON. OpenFold3
#: derives its sampling from these, so the same query gives the same prediction
#: (up to GPU non-determinism). The value carries no meaning; pass ``seeds=`` to
#: the ``prepare_*`` and ``run_openfold_*`` functions to use others.
_DEFAULT_QUERY_SEEDS: tuple[int, ...] = (42,)


def _query_seeds(seeds: Sequence[int]) -> list[int]:
    """Validate ``seeds`` and return them as a list of ints for the query JSON."""
    if isinstance(seeds, (str, bytes)):
        raise TypeError("seeds must be a sequence of integers, not a string.")
    values = [int(s) for s in seeds]
    if not values:
        raise ValueError("seeds must contain at least one integer.")
    return values


def _write_runner_yaml(
    output_dir: Path,
    presets: list[str],
    template_dir: Optional[Path] = None,
) -> Path:
    """Write a runner YAML with model presets and optional template settings.

    Args:
        output_dir: Directory in which to write the file.
        presets: List of model preset names, e.g.
            ``["predict", "pae_enabled", "low_mem"]``.
        template_dir: If given, adds ``template_preprocessor_settings`` with
            ``structure_directory`` pointing here and
            ``fetch_missing_structures: false`` so OF3 uses local CIFs only.

    Returns:
        Path to the written YAML file.
    """
    cfg: dict = {"model_update": {"presets": presets}}
    if template_dir is not None:
        cfg["template_preprocessor_settings"] = {
            "structure_directory": str(template_dir),
            "structure_file_format": "cif",
            "fetch_missing_structures": False,
        }

    try:
        import yaml

        content = yaml.dump(cfg, default_flow_style=False)
    except ImportError:
        # Fallback: write YAML manually
        lines = ["model_update:\n", "  presets:\n"]
        lines += [f"    - {p}\n" for p in presets]
        if template_dir is not None:
            lines += [
                "template_preprocessor_settings:\n",
                f"  structure_directory: {template_dir}\n",
                "  structure_file_format: cif\n",
                "  fetch_missing_structures: false\n",
            ]
        content = "".join(lines)

    yaml_path = output_dir / "runner_config.yaml"
    yaml_path.write_text(content)
    return yaml_path


def _extract_sequence_from_structure(structure, chain_id: str) -> str:
    """Extract one-letter amino acid sequence for a chain using gemmi.

    Skips non-amino-acid residues (waters, ligands, etc.).

    Args:
        structure: A ``gemmi.Structure`` object.
        chain_id: Chain ID to extract.

    Returns:
        One-letter sequence string (non-standard residues become 'X').

    Raises:
        ValueError: If the chain is not found or contains no amino acids.
    """
    import gemmi

    for model in structure:
        for chain in model:
            if chain.name != chain_id:
                continue
            seq = []
            for res in chain:
                tbl = gemmi.find_tabulated_residue(res.name)
                if tbl is None or not tbl.is_amino_acid():
                    continue
                seq.append(tbl.one_letter_code or "X")
            if not seq:
                raise ValueError(f"Chain '{chain_id}' found but contains no amino acid residues")
            return "".join(seq)
    raise ValueError(f"Chain '{chain_id}' not found in structure")


def _extract_chain_to_cif(structure, chain_id: str, output_path: Path, sequence: str = "") -> None:
    """Write a single chain from a gemmi Structure to a CIF file.

    Also patches the missing mmCIF metadata tables required by OF3's template
    preprocessor (``_pdbx_audit_revision_history``, ``_entity_poly``,
    ``_pdbx_poly_seq_scheme``).

    Args:
        structure: Source ``gemmi.Structure``.
        chain_id: Chain ID to extract (taken from the first model).
        output_path: Destination CIF file path.
        sequence: One-letter amino acid sequence for this chain (used to
            populate ``_entity_poly``). If empty, extracted from structure atoms.
    """
    import gemmi

    new_st = gemmi.Structure()
    new_st.cell = structure.cell
    new_st.spacegroup_hm = structure.spacegroup_hm
    new_model = gemmi.Model("1")
    for chain in structure[0]:
        if chain.name == chain_id:
            new_model.add_chain(chain.clone())
            break
    new_st.add_model(new_model)
    new_st.make_mmcif_document().write_file(str(output_path))

    # OF3 template preprocessor requires metadata tables that OpenMM/gemmi
    # do not write. Patch them in using gemmi's CIF API.
    if not sequence:
        sequence = _extract_sequence_from_structure(new_st, chain_id)

    doc = gemmi.cif.read(str(output_path))
    block = doc.sole_block()

    # 1. Release date — OF3 requires this; use a sentinel far-past date so
    #    no template release-date filter will discard it.
    rdh = block.init_loop("_pdbx_audit_revision_history.", ["ordinal", "revision_date"])
    rdh.add_row(["1", "1900-01-01"])

    # 2. entity_poly — entity_id → canonical 1-letter sequence.
    #    gemmi's make_mmcif_document() writes entity_id as "?" in struct_asym,
    #    so we always use "1" (single-chain, single-entity template).
    entity_id = "1"
    ep = block.init_loop("_entity_poly.", ["entity_id", "pdbx_seq_one_letter_code_can"])
    ep.add_row([entity_id, sequence])

    # 3. pdbx_poly_seq_scheme — asym_id → entity_id.
    #    Materialise struct_asym rows eagerly before init_loop (block.find()
    #    returns a lazy view that is invalidated by subsequent init_loop calls).
    sa = block.find(["_struct_asym.id"])
    asym_ids = [row[0] for row in sa] if sa else [chain_id]
    pss = block.init_loop("_pdbx_poly_seq_scheme.", ["asym_id", "entity_id"])
    for asym_id in asym_ids:
        pss.add_row([asym_id, entity_id])

    doc.write_file(str(output_path))


def _write_a3m_self_alignment(
    sequence: str,
    query_id: str,
    entry_id: str,
    chain_id: str,
    output_path: Path,
) -> None:
    """Write a minimal A3M alignment for a self-template (100% identity).

    The A3M contains two sequences: the query and the template (identical).
    Header format follows OF3's A3M parser::

        >{entry_id}_{chain_id}/{start}-{end}

    OF3 splits on ``_`` to get (entry_id, chain_id), then looks for
    ``{entry_id}.cif`` in the ``template_preprocessor_settings.structure_directory``.

    Args:
        sequence: One-letter amino acid sequence.
        query_id: Identifier for the query (first) sequence.
        entry_id: Template entry identifier — must contain no underscores.
            OF3 looks for ``{entry_id}.cif`` in the template directory.
        chain_id: Chain identifier within the template CIF (e.g. ``"A"``).
        output_path: Destination A3M file path.
    """
    n = len(sequence)
    template_header = f"{entry_id}_{chain_id}/{1}-{n}"
    output_path.write_text(f">{query_id}/1-{n}\n{sequence}\n>{template_header}\n{sequence}\n")


def prepare_refolding_query(
    complex_structure_path: str | Path,
    receptor_chain: str,
    binder_chain: str,
    query_name: str,
    output_dir: str | Path,
    template_cif_path: Optional[str | Path] = None,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
) -> Path:
    """Prepare an OpenFold3 query JSON for binder refolding with receptor as template.

    **Mode 2 — target-fixed refolding:**
    The receptor chain is provided as a structural template so that OF3 is
    conditioned on the known receptor geometry. The binder chain is predicted
    from sequence only (no template). This answers: "given the known receptor,
    can OF3 recover the bound binder conformation?"

    Files written under ``output_dir``:

    .. code-block:: text

        {output_dir}/
          {query_name}_query.json            — OF3 input JSON
          {query_name}_receptor_{chain}.a3m  — self-alignment for receptor
          templates/
            {query_name}_receptor_{chain}.cif — receptor template structure

    The template CIF must be discoverable by OF3 at inference time. Pass::

        --template_mmcif_dir {output_dir}/templates

    to ``run_openfold`` (via ``extra_args``) if OF3 does not automatically
    locate templates relative to the alignment file.

    .. note::
        OF3's template pipeline currently supports **monomeric templates**
        (protein chains only). The extracted receptor CIF is a single-chain
        structure; multi-chain receptors should be merged into one chain
        before calling this function, or handled with separate templates per
        chain.

    Args:
        complex_structure_path: CIF or PDB file of the full complex.
        receptor_chain: Chain ID of the receptor/target to fix as template.
        binder_chain: Chain ID of the binder to refold (no template).
        query_name: Name for the prediction query (used in file names and the
            OF3 ``name`` field).
        output_dir: Directory to write query JSON and supporting files.
        template_cif_path: Optional path to a pre-prepared receptor CIF
            (e.g., after MD relaxation). Must be a monomer (one chain).
            If None, the receptor chain is extracted from
            ``complex_structure_path``.
        seeds: Seed values written to the query JSON's ``"seeds"`` field
            (default ``(42,)``). One prediction is made per seed.

    Returns:
        Path to the written query JSON file.

    Raises:
        ValueError: If a specified chain is not found or has no amino acids,
            or ``seeds`` is empty.
    """
    import gemmi

    seed_values = _query_seeds(seeds)
    complex_structure_path = Path(complex_structure_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = output_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    # Extract sequences
    st = gemmi.read_structure(str(complex_structure_path))
    receptor_seq = _extract_sequence_from_structure(st, receptor_chain)
    binder_seq = _extract_sequence_from_structure(st, binder_chain)

    # Template CIF — named receptor.cif so OF3 finds entry_id="receptor"
    receptor_entry_id = "receptor"
    template_dest = templates_dir / f"{receptor_entry_id}.cif"
    if template_cif_path is not None:
        # Extract only the receptor chain from the provided CIF
        template_src = gemmi.read_structure(str(template_cif_path))
        _extract_chain_to_cif(template_src, receptor_chain, template_dest, sequence=receptor_seq)
    else:
        _extract_chain_to_cif(st, receptor_chain, template_dest, sequence=receptor_seq)

    # A3M self-alignment for receptor — header: receptor_{chain}/{1}-{N}
    a3m_path = output_dir / f"{query_name}_receptor.a3m"
    _write_a3m_self_alignment(
        receptor_seq,
        f"query_{receptor_chain}",
        receptor_entry_id,
        receptor_chain,
        a3m_path,
    )

    # Query JSON — receptor has template, binder is free (sequence only)
    query = {
        "seeds": seed_values,
        "queries": {
            query_name: {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": [receptor_chain],
                        "sequence": receptor_seq,
                        "template_alignment_file_path": str(a3m_path),
                    },
                    {
                        "molecule_type": "protein",
                        "chain_ids": [binder_chain],
                        "sequence": binder_seq,
                    },
                ],
            }
        },
    }
    query_json_path = output_dir / f"{query_name}_query.json"
    query_json_path.write_text(json.dumps(query, indent=2))
    return query_json_path


def prepare_scoring_query(
    complex_structure_path: str | Path,
    receptor_chain: str,
    binder_chain: str,
    query_name: str,
    output_dir: str | Path,
    template_cif_path: Optional[str | Path] = None,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
) -> Path:
    """Prepare an OpenFold3 query JSON to score an existing complex structure.

    **Mode 1 — structure scoring:**
    Both receptor and binder chains are provided as structural templates so
    that OF3 evaluates the known conformation rather than predicting de-novo.
    OF3 outputs confidence scores (pLDDT, pTM, ipTM, etc.) reflecting how
    self-consistent it finds that specific structure.

    Files written under ``output_dir``:

    .. code-block:: text

        {output_dir}/
          {query_name}_query.json
          {query_name}_receptor_{chain}.a3m
          {query_name}_binder_{chain}.a3m
          templates/
            {query_name}_receptor_{chain}.cif
            {query_name}_binder_{chain}.cif

    Args:
        complex_structure_path: CIF or PDB file of the full complex.
        receptor_chain: Chain ID of the receptor.
        binder_chain: Chain ID of the binder.
        query_name: Name for the prediction query.
        output_dir: Directory to write query JSON and supporting files.
        template_cif_path: Optional pre-prepared complex or receptor CIF
            (e.g., after MD relaxation). When provided, both chain templates
            are extracted from this file instead of ``complex_structure_path``.
        seeds: Seed values written to the query JSON's ``"seeds"`` field
            (default ``(42,)``). One prediction is made per seed.

    Returns:
        Path to the written query JSON file.

    Raises:
        ValueError: If ``seeds`` is empty.
    """
    import gemmi

    seed_values = _query_seeds(seeds)
    complex_structure_path = Path(complex_structure_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = output_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    # Source structure for sequences (always the original complex)
    st = gemmi.read_structure(str(complex_structure_path))
    receptor_seq = _extract_sequence_from_structure(st, receptor_chain)
    binder_seq = _extract_sequence_from_structure(st, binder_chain)

    # Template source: use template_cif_path if provided, else the complex
    template_src = (
        gemmi.read_structure(str(template_cif_path)) if template_cif_path is not None else st
    )

    # Template CIF files — named {entry_id}.cif so OF3 can find them.
    # Entry IDs must contain no underscores; chain_id is the suffix after "_".
    receptor_entry_id = "receptor"
    binder_entry_id = "binder"

    _extract_chain_to_cif(
        template_src,
        receptor_chain,
        templates_dir / f"{receptor_entry_id}.cif",
        sequence=receptor_seq,
    )
    _extract_chain_to_cif(
        template_src, binder_chain, templates_dir / f"{binder_entry_id}.cif", sequence=binder_seq
    )

    # A3M self-alignments — header: {entry_id}_{chain_id}/{1}-{N}
    receptor_a3m = output_dir / f"{query_name}_receptor.a3m"
    binder_a3m = output_dir / f"{query_name}_binder.a3m"
    _write_a3m_self_alignment(
        receptor_seq,
        f"query_{receptor_chain}",
        receptor_entry_id,
        receptor_chain,
        receptor_a3m,
    )
    _write_a3m_self_alignment(
        binder_seq,
        f"query_{binder_chain}",
        binder_entry_id,
        binder_chain,
        binder_a3m,
    )

    # Query JSON — OF3 format: {"seeds": [...], "queries": {"name": {"chains": [...]}}}
    query = {
        "seeds": seed_values,
        "queries": {
            query_name: {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": [receptor_chain],
                        "sequence": receptor_seq,
                        "template_alignment_file_path": str(receptor_a3m),
                    },
                    {
                        "molecule_type": "protein",
                        "chain_ids": [binder_chain],
                        "sequence": binder_seq,
                        "template_alignment_file_path": str(binder_a3m),
                    },
                ],
            }
        },
    }
    query_json_path = output_dir / f"{query_name}_query.json"
    query_json_path.write_text(json.dumps(query, indent=2))
    return query_json_path


@dataclasses.dataclass
class _BatchSample:
    """Descriptor for one sample in a batched OF3 run."""

    query_name: str
    complex_structure_path: Path
    receptor_chain: str
    binder_chain: str


def _safe_entry_id(sample_id: str, suffix: str) -> str:
    """Return an OF3-safe entry ID (no underscores).

    OF3 splits A3M template headers on ``_`` to get (entry_id, chain_id),
    so entry_id must not contain underscores.
    """
    return sample_id.replace("_", "-") + suffix


def prepare_batched_scoring_queries(
    samples: list[_BatchSample],
    output_dir: str | Path,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
) -> Path:
    """Prepare a single OF3 query JSON that scores multiple complexes.

    All template CIFs and A3M files are written into a shared directory
    structure so that one ``run_openfold predict`` call processes every
    sample.

    Args:
        samples: Per-sample descriptors.
        output_dir: Directory to write query JSON and supporting files.
        seeds: Seed values written to the query JSON (default ``(42,)``).

    Returns:
        Path to the combined query JSON file.

    Raises:
        ValueError: If ``seeds`` is empty.
    """
    import gemmi

    seed_values = _query_seeds(seeds)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = output_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    queries: dict = {}
    for s in samples:
        st = gemmi.read_structure(str(s.complex_structure_path))
        receptor_seq = _extract_sequence_from_structure(st, s.receptor_chain)
        binder_seq = _extract_sequence_from_structure(st, s.binder_chain)

        rec_entry = _safe_entry_id(s.query_name, "rec")
        bnd_entry = _safe_entry_id(s.query_name, "bnd")

        _extract_chain_to_cif(
            st, s.receptor_chain, templates_dir / f"{rec_entry}.cif", sequence=receptor_seq
        )
        _extract_chain_to_cif(
            st, s.binder_chain, templates_dir / f"{bnd_entry}.cif", sequence=binder_seq
        )

        rec_a3m = output_dir / f"{s.query_name}_receptor.a3m"
        bnd_a3m = output_dir / f"{s.query_name}_binder.a3m"
        _write_a3m_self_alignment(
            receptor_seq, f"query_{s.receptor_chain}", rec_entry, s.receptor_chain, rec_a3m
        )
        _write_a3m_self_alignment(
            binder_seq, f"query_{s.binder_chain}", bnd_entry, s.binder_chain, bnd_a3m
        )

        queries[s.query_name] = {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": [s.receptor_chain],
                    "sequence": receptor_seq,
                    "template_alignment_file_path": str(rec_a3m),
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": [s.binder_chain],
                    "sequence": binder_seq,
                    "template_alignment_file_path": str(bnd_a3m),
                },
            ],
        }

    query_json_path = output_dir / "batch_query.json"
    query_json_path.write_text(json.dumps({"seeds": seed_values, "queries": queries}, indent=2))
    return query_json_path


def prepare_batched_refolding_queries(
    samples: list[_BatchSample],
    output_dir: str | Path,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
) -> Path:
    """Prepare a single OF3 query JSON that refolds binders for multiple complexes.

    Receptor chains are provided as structural templates; binder chains are
    predicted from sequence only.

    Args:
        samples: Per-sample descriptors.
        output_dir: Directory to write query JSON and supporting files.
        seeds: Seed values written to the query JSON (default ``(42,)``).

    Returns:
        Path to the combined query JSON file.

    Raises:
        ValueError: If ``seeds`` is empty.
    """
    import gemmi

    seed_values = _query_seeds(seeds)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = output_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    queries: dict = {}
    for s in samples:
        st = gemmi.read_structure(str(s.complex_structure_path))
        receptor_seq = _extract_sequence_from_structure(st, s.receptor_chain)
        binder_seq = _extract_sequence_from_structure(st, s.binder_chain)

        rec_entry = _safe_entry_id(s.query_name, "rec")

        _extract_chain_to_cif(
            st, s.receptor_chain, templates_dir / f"{rec_entry}.cif", sequence=receptor_seq
        )

        rec_a3m = output_dir / f"{s.query_name}_receptor.a3m"
        _write_a3m_self_alignment(
            receptor_seq, f"query_{s.receptor_chain}", rec_entry, s.receptor_chain, rec_a3m
        )

        queries[s.query_name] = {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": [s.receptor_chain],
                    "sequence": receptor_seq,
                    "template_alignment_file_path": str(rec_a3m),
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": [s.binder_chain],
                    "sequence": binder_seq,
                },
            ],
        }

    query_json_path = output_dir / "batch_query.json"
    query_json_path.write_text(json.dumps({"seeds": seed_values, "queries": queries}, indent=2))
    return query_json_path
