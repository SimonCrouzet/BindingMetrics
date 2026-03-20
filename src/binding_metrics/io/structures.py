"""Structure loading and manipulation utilities."""

import tempfile
import warnings
from pathlib import Path
from typing import Optional

from openmm import app
from openmm.app import PDBFile


def load_complex(pdb_path: str | Path) -> PDBFile:
    """Load a PDB file containing a molecular complex.

    Args:
        pdb_path: Path to the PDB file

    Returns:
        Loaded PDBFile object

    Raises:
        FileNotFoundError: If the PDB file doesn't exist
        ValueError: If the file cannot be parsed
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    try:
        return PDBFile(str(pdb_path))
    except Exception as e:
        raise ValueError(f"Failed to parse PDB file: {e}") from e


def get_chain_atom_indices(
    pdb_path: str | Path,
    chain_ids: list[str],
) -> list[int]:
    """Get atom indices for specified chains.

    Args:
        pdb_path: Path to PDB file
        chain_ids: List of chain IDs to extract

    Returns:
        List of atom indices (0-based) belonging to the specified chains
    """
    pdb = PDBFile(str(pdb_path))
    topology = pdb.topology

    indices = []
    for atom in topology.atoms():
        if atom.residue.chain.id in chain_ids:
            indices.append(atom.index)

    return indices


def load_structure(path: str | Path) -> tuple:
    """Load a structure file (PDB or CIF) and return (topology, positions).

    Supports .pdb, .cif, and .mmcif formats.

    Args:
        path: Path to the structure file

    Returns:
        Tuple of (topology, positions) as OpenMM objects

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the format is unsupported or parsing fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix in (".cif", ".mmcif"):
            from openmm.app import PDBxFile
            struct = PDBxFile(str(path))
        elif suffix == ".pdb":
            struct = PDBFile(str(path))
        else:
            raise ValueError(f"Unsupported structure format: {suffix}. Use .pdb, .cif, or .mmcif")
    except (ValueError, FileNotFoundError):
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse structure file {path}: {e}") from e

    return struct.topology, struct.positions


def detect_chains(topology) -> tuple[Optional[str], Optional[str]]:
    """Auto-detect ligand (peptide) and receptor chain IDs from topology.

    Identifies protein chains by counting standard amino acid residues.
    Returns the smallest chain as ligand and largest as receptor.
    If only one chain exists, returns it as ligand and None as receptor.

    Args:
        topology: OpenMM Topology object

    Returns:
        Tuple of (ligand_chain_id, receptor_chain_id). Either may be None
        if no protein chains are found or only one chain exists.
    """
    # Amino acids only — exclude water (HOH) and nucleic acids which are also
    # in app.PDBFile._standardResidues and would cause water chains to be ranked.
    amino_acids = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL",
        "CYX", "HID", "HIE", "HIP", "ASPL", "GLUL", "LYSL",
        "NMG", "NMA", "MVA", "MLE",
    }

    chain_sizes = []
    for chain in topology.chains():
        n_protein = sum(1 for r in chain.residues() if r.name in amino_acids)
        if n_protein > 0:
            chain_sizes.append((chain.id, n_protein))

    if not chain_sizes:
        return None, None

    chain_sizes.sort(key=lambda x: x[1])

    if len(chain_sizes) == 1:
        return chain_sizes[0][0], None

    return chain_sizes[0][0], chain_sizes[-1][0]


def strip_heterogens(
    topology,
    positions,
    peptide_chain: Optional[str],
    receptor_chain: Optional[str],
    warn_cutoff_ang: float = 8.0,
):
    """Remove non-protein residues from topology, warning if close to the interface.

    Args:
        topology: OpenMM Topology (post-PDBFixer).
        positions: Atom positions (OpenMM Quantity, nm).
        peptide_chain: Peptide chain ID to preserve.
        receptor_chain: Receptor chain ID to preserve.
        warn_cutoff_ang: Distance threshold in Å; heterogens within this distance
            trigger a warning before removal.

    Returns:
        Tuple (topology, positions) with heterogens removed.
    """
    import numpy as np

    amino_acids = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL",
        "CYX", "HID", "HIE", "HIP", "ASPL", "GLUL", "LYSL",
        "NMG", "NMA", "MVA", "MLE",
    }

    protein_chain_ids = {c for c in (peptide_chain, receptor_chain) if c}
    protein_pos = np.array([
        [p.x, p.y, p.z]
        for a, p in zip(topology.atoms(), positions)
        if a.residue.chain.id in protein_chain_ids
    ]) * 10  # nm → Å

    _water_names = {"HOH", "WAT", "TIP", "TIP3", "SOL"}

    atoms_to_remove = []
    for res in topology.residues():
        if res.chain.id in protein_chain_ids:
            continue
        if res.name in amino_acids:
            continue
        # Water: always remove silently
        if res.name in _water_names:
            atoms_to_remove.extend(res.atoms())
            continue
        # Other heterogens: warn if close to protein (may be a cofactor/ion)
        res_pos = np.array([
            [positions[a.index].x, positions[a.index].y, positions[a.index].z]
            for a in res.atoms()
        ]) * 10  # nm → Å
        if len(protein_pos) > 0 and len(res_pos) > 0:
            dists = np.linalg.norm(
                res_pos[:, None, :] - protein_pos[None, :, :], axis=-1
            )
            min_dist = float(dists.min())
            if min_dist < warn_cutoff_ang:
                print(f"  Warning: removing heterogen {res.name}{res.id} "
                      f"(chain {res.chain.id}) which is {min_dist:.1f} Å from "
                      f"the protein — it may be a functional cofactor or ion. "
                      f"Parametrize it via custom_bond_handler to keep it.")
            else:
                print(f"  Removing distant heterogen {res.name}{res.id} "
                      f"(chain {res.chain.id}, {min_dist:.1f} Å from protein)")
        else:
            print(f"  Removing heterogen {res.name}{res.id} (chain {res.chain.id})")
        atoms_to_remove.extend(res.atoms())

    if atoms_to_remove:
        modeller = app.Modeller(topology, positions)
        modeller.delete(atoms_to_remove)
        topology, positions = modeller.topology, modeller.positions

    return topology, positions


def _patch_nonstd_bonds_in_cif(cif_path: Path, topology) -> None:
    """Add non-disulfide non-sequential intra-chain bonds to _struct_conn.

    PDBxFile.writeFile only records disulfide bonds.  Custom covalent bonds
    (head-to-tail amide, lactam, etc.) are silently omitted, so PDBxFile
    cannot round-trip them.  This function reads the written CIF back with
    gemmi and appends the missing covale rows.
    """
    try:
        import gemmi
    except ImportError:
        return

    # Map global residue index → local 1-based position within its chain
    res_local: dict[int, int] = {}
    for chain in topology.chains():
        for local_idx, res in enumerate(chain.residues(), start=1):
            res_local[res.index] = local_idx

    custom_bonds = []
    for bond in topology.bonds():
        a1, a2 = bond.atom1, bond.atom2
        r1, r2 = a1.residue, a2.residue
        if r1.chain.id != r2.chain.id:
            continue
        if abs(r1.index - r2.index) <= 1:
            continue
        if a1.name == "SG" and a2.name == "SG":
            continue  # disulfide, already in _struct_conn
        custom_bonds.append((a1, a2))

    if not custom_bonds:
        return

    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()

    existing = block.find(["_struct_conn.id"])
    n_existing = len(existing) if existing else 0

    _STRUCT_CONN_COLS = [
        "_struct_conn.id",
        "_struct_conn.conn_type_id",
        "_struct_conn.ptnr1_auth_asym_id",
        "_struct_conn.ptnr1_label_comp_id",
        "_struct_conn.ptnr1_auth_seq_id",
        "_struct_conn.ptnr1_label_atom_id",
        "_struct_conn.ptnr2_auth_asym_id",
        "_struct_conn.ptnr2_label_comp_id",
        "_struct_conn.ptnr2_auth_seq_id",
        "_struct_conn.ptnr2_label_atom_id",
    ]
    loop_ref = block.find_loop("_struct_conn.id")
    if loop_ref:
        loop = loop_ref.get_loop()
    else:
        loop = block.init_loop("_struct_conn.", [col.split(".")[1] for col in _STRUCT_CONN_COLS])

    for i, (a1, a2) in enumerate(custom_bonds):
        r1, r2 = a1.residue, a2.residue
        bond_id = f"covale{n_existing + i + 1}"
        loop.add_row([
            bond_id, "covale",
            r1.chain.id, r1.name, str(res_local[r1.index]), a1.name,
            r2.chain.id, r2.name, str(res_local[r2.index]), a2.name,
        ])

    doc.write_file(str(cif_path))


def save_cif(
    topology,
    positions,
    output_path: str | Path,
    source_cif_path: Optional[str | Path] = None,
) -> None:
    """Save structure as a CIF file.

    If source_cif_path is provided, coordinates are merged into the source CIF
    to preserve metadata (entity descriptions, sequence info, etc.).
    Falls back to raw OpenMM CIF output if gemmi is not available.

    Args:
        topology: OpenMM Topology object
        positions: OpenMM positions
        output_path: Path to write the output CIF file
        source_cif_path: Optional source CIF whose metadata will be preserved
    """
    from openmm.app import PDBxFile

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if source_cif_path is None:
        with open(output_path, "w") as f:
            PDBxFile.writeFile(topology, positions, f)
        # PDBxFile only writes disulfide bonds to _struct_conn; patch in any
        # other non-sequential intra-chain covalent bonds (e.g. head-to-tail).
        _patch_nonstd_bonds_in_cif(output_path, topology)
        return

    try:
        import gemmi
    except ImportError:
        with open(output_path, "w") as f:
            PDBxFile.writeFile(topology, positions, f)
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        PDBxFile.writeFile(topology, positions, tmp)

    try:
        source_doc = gemmi.cif.read(str(source_cif_path))
        source_block = source_doc.sole_block()

        update_doc = gemmi.cif.read(str(tmp_path))
        update_block = update_doc.sole_block()

        # Build coordinate lookup from OpenMM output
        update_coords: dict = {}
        update_loop = update_block.find(
            "_atom_site.",
            ["auth_asym_id", "auth_seq_id", "auth_atom_id", "Cartn_x", "Cartn_y", "Cartn_z"],
        )
        if update_loop:
            for row in update_loop:
                key = (row[0], row[1], row[2])
                update_coords[key] = (row[3], row[4], row[5])

        # Validate coordinate counts before merging
        source_table = source_block.find(
            "_atom_site.",
            ["auth_asym_id", "auth_seq_id", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z"],
        )
        if source_table and update_coords:
            n_source = len(source_table)
            n_update = len(update_coords)
            if n_source != n_update:
                warnings.warn(
                    f"save_cif: coordinate count mismatch — source CIF has {n_source} "
                    f"_atom_site rows but OpenMM output has {n_update} atoms. "
                    "Some coordinates may not be updated.",
                    stacklevel=2,
                )
            used_keys: set = set()
            for row in source_table:
                key = (row[0], row[1], row[2])
                if key in update_coords:
                    row[3], row[4], row[5] = update_coords[key]
                    used_keys.add(key)
            unused_keys = set(update_coords.keys()) - used_keys
            if unused_keys:
                warnings.warn(
                    f"save_cif: {len(unused_keys)} atom(s) from OpenMM output were not found "
                    "in the source CIF _atom_site and were not merged. "
                    f"Example unmatched key: {next(iter(unused_keys))}",
                    stacklevel=2,
                )

        source_doc.write_file(str(output_path))
    finally:
        tmp_path.unlink(missing_ok=True)


def save_structure(
    topology,
    positions,
    output_path: str | Path,
    source_path: Optional[str | Path] = None,
) -> None:
    """Save structure as PDB or CIF based on the output file extension.

    Args:
        topology: OpenMM Topology object
        positions: OpenMM positions
        output_path: Destination file (.pdb, .cif, or .mmcif)
        source_path: Optional source file; if CIF, its metadata is preserved in output
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix in (".cif", ".mmcif"):
        src = (
            source_path
            if source_path and Path(source_path).suffix.lower() in (".cif", ".mmcif")
            else None
        )
        save_cif(topology, positions, output_path, source_cif_path=src)
    elif suffix == ".pdb":
        with open(output_path, "w") as f:
            PDBFile.writeFile(topology, positions, f)
    else:
        raise ValueError(f"Unsupported output format: {suffix!r}. Use .pdb, .cif, or .mmcif")


def get_residue_info(pdb_path: str | Path) -> list[dict]:
    """Get information about residues in the structure.

    Args:
        pdb_path: Path to PDB file

    Returns:
        List of dictionaries with residue information
    """
    pdb = PDBFile(str(pdb_path))
    topology = pdb.topology

    residues = []
    for residue in topology.residues():
        residues.append({
            "name": residue.name,
            "index": residue.index,
            "chain": residue.chain.id,
            "n_atoms": len(list(residue.atoms())),
        })

    return residues
