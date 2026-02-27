"""Structure loading and manipulation utilities."""

import tempfile
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
    standard_residues = set(app.PDBFile._standardResidues)

    chain_sizes = []
    for chain in topology.chains():
        n_protein = sum(1 for r in chain.residues() if r.name in standard_residues)
        if n_protein > 0:
            chain_sizes.append((chain.id, n_protein))

    if not chain_sizes:
        return None, None

    chain_sizes.sort(key=lambda x: x[1])

    if len(chain_sizes) == 1:
        return chain_sizes[0][0], None

    return chain_sizes[0][0], chain_sizes[-1][0]


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

        # Update coordinates in source block
        source_table = source_block.find(
            "_atom_site.",
            ["auth_asym_id", "auth_seq_id", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z"],
        )
        if source_table and update_coords:
            for row in source_table:
                key = (row[0], row[1], row[2])
                if key in update_coords:
                    row[3], row[4], row[5] = update_coords[key]

        source_doc.write_file(str(output_path))
    finally:
        tmp_path.unlink(missing_ok=True)


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
