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


def detect_chains_from_file(
    path,
    peptide_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Detect or confirm peptide and receptor chain IDs from a structure file.

    Uses biotite for fast chain analysis. When chain IDs are not provided,
    the smallest protein chain is assigned as peptide and the largest as receptor.
    When they are provided, the function just confirms and reports their properties.

    Args:
        path: Path to CIF or PDB file.
        peptide_chain: Explicit peptide chain ID, or None to auto-detect.
        receptor_chain: Explicit receptor chain ID, or None to auto-detect.
        verbose: If True, print detected chain info to stdout.

    Returns:
        Dict with keys:
            peptide_chain (str): resolved peptide chain ID
            receptor_chain (str): resolved receptor chain ID
            peptide_n_residues (int): number of residues in peptide chain
            receptor_n_residues (int): number of residues in receptor chain
            all_chains (list[dict]): all protein chains with id and n_residues
    """
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure.io.pdb as pdb_io

    path = Path(path)
    suffix = path.suffix.lower()

    # Build auth→label chain ID mapping for CIF files.
    # biotite uses auth_asym_id by default; OpenMM uses label_asym_id.
    # When they differ, OpenMM-based steps need the label ID.
    # We restrict the mapping to CA atoms so that ligand/water label chains
    # (which share the same auth chain as the protein) don't overwrite the
    # protein label.
    auth_to_label: dict[str, str] = {}
    if suffix in (".cif", ".mmcif"):
        f = pdbx.CIFFile.read(str(path))
        atoms = pdbx.get_structure(f, model=1)
        try:
            atom_site = f.block["atom_site"]
            auth_ids = atom_site["auth_asym_id"].as_array()
            label_ids = atom_site["label_asym_id"].as_array()
            atom_names = atom_site["auth_atom_id"].as_array()
            for a, l, name in zip(auth_ids, label_ids, atom_names):
                if str(name).strip() == "CA":   # protein Cα only
                    a_str, l_str = str(a), str(l)
                    if a_str not in auth_to_label:   # first occurrence wins
                        auth_to_label[a_str] = l_str
        except Exception:
            pass  # not all CIF files have both columns; mapping stays empty
    else:
        f = pdb_io.PDBFile.read(str(path))
        atoms = pdb_io.get_structure(f, model=1)

    # Count standard amino-acid residues per chain
    aa_filter = struc.filter_amino_acids(atoms)
    aa_atoms = atoms[aa_filter]
    chain_ids = sorted(set(aa_atoms.chain_id))

    chain_info = []
    for cid in chain_ids:
        n_res = len(set(zip(
            aa_atoms.chain_id[aa_atoms.chain_id == cid],
            aa_atoms.res_id[aa_atoms.chain_id == cid],
        )))
        chain_info.append({"id": str(cid), "n_residues": int(n_res)})

    chain_info.sort(key=lambda c: c["n_residues"])

    if not chain_info:
        raise ValueError(f"No protein chains found in {path}")

    # Resolve IDs
    if peptide_chain is None:
        peptide_chain = chain_info[0]["id"]
        pep_auto = True
    else:
        pep_auto = False

    if receptor_chain is None and len(chain_info) > 1:
        if len(chain_info) == 2:
            receptor_chain = chain_info[1]["id"]
        else:
            # Multiple candidates — pick the one with the most Cα contacts
            # to the peptide within 8 Å (interface-proximity criterion).
            pep_ca = aa_atoms[
                (aa_atoms.chain_id == peptide_chain) &
                (aa_atoms.atom_name == "CA")
            ]
            best_chain, best_contacts = None, -1
            for ci in chain_info:
                if ci["id"] == peptide_chain:
                    continue
                cand_ca = aa_atoms[
                    (aa_atoms.chain_id == ci["id"]) &
                    (aa_atoms.atom_name == "CA")
                ]
                if len(pep_ca) == 0 or len(cand_ca) == 0:
                    contacts = 0
                else:
                    from biotite.structure import distance
                    import numpy as np
                    dists = np.array([
                        distance(pep_ca.coord, cand_ca.coord[j]).min()
                        for j in range(len(cand_ca))
                    ])
                    contacts = int((dists < 8.0).sum())
                if contacts > best_contacts:
                    best_contacts, best_chain = contacts, ci["id"]
            receptor_chain = best_chain
        rec_auto = True
    else:
        rec_auto = receptor_chain is None

    pep_info = next((c for c in chain_info if c["id"] == peptide_chain), None)
    rec_info = next((c for c in chain_info if c["id"] == receptor_chain), None) if receptor_chain else None

    if verbose:
        print(f"  Chain detection ({path.name}):")
        for c in chain_info:
            tag = ""
            if c["id"] == peptide_chain:
                tag = "  ← peptide" + (" [auto]" if pep_auto else "")
            elif c["id"] == receptor_chain:
                tag = "  ← receptor" + (" [auto]" if rec_auto else "")
            print(f"    chain {c['id']}: {c['n_residues']} residues{tag}")

    pep_str = str(peptide_chain) if peptide_chain else None
    rec_str = str(receptor_chain) if receptor_chain else None
    return {
        "peptide_chain": pep_str,
        "receptor_chain": rec_str,
        # label_asym_id equivalents for OpenMM-based steps (same as auth when
        # both ID systems are identical, i.e. standard PDB structures)
        "peptide_chain_label": auth_to_label.get(pep_str, pep_str) if pep_str else None,
        "receptor_chain_label": auth_to_label.get(rec_str, rec_str) if rec_str else None,
        "peptide_n_residues": pep_info["n_residues"] if pep_info else None,
        "receptor_n_residues": rec_info["n_residues"] if rec_info else None,
        "all_chains": chain_info,
    }


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

    Writes a fresh CIF from OpenMM (correct atoms, hydrogens, bonds), then
    patches the auth_asym_id and auth_seq_id columns back to the original
    values from source_cif_path so that chain IDs and residue numbers are
    preserved.

    OpenMM uses label_asym_id internally and resets auth_seq_id to 1-based
    sequential integers.  A single source auth chain may also be split into
    multiple label chains by OpenMM.  The patch step handles both by building
    a label→auth mapping and a positional (auth_chain, res_idx) → auth_seq_id
    table from the source CIF.

    Falls back to raw OpenMM output if gemmi is not available or source is None.

    Args:
        topology: OpenMM Topology object
        positions: OpenMM positions
        output_path: Path to write the output CIF file
        source_cif_path: Optional source CIF used to restore original auth IDs
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

    # Write fresh OpenMM CIF — correct atoms and H, but with label chain IDs
    # and 1-based sequential auth_seq_id.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        PDBxFile.writeFile(topology, positions, tmp)

    try:
        source_block = gemmi.cif.read(str(source_cif_path)).sole_block()
        output_doc   = gemmi.cif.read(str(tmp_path))
        output_block = output_doc.sole_block()

        # ── Restore original chain IDs and residue numbers ───────────────────────
        #
        # PDBxFile.writeFile assigns sequential letters (A, B, C…) and resets
        # residue numbers to 1-based, ignoring the topology's chain IDs.
        # The topology carries the source label_asym_id as chain IDs (that's what
        # PDBxFile sets when it reads a CIF).  So we need two things from the source:
        #
        #   label_to_auth : source label_asym_id → source auth_asym_id
        #   seq_map       : (source_auth_chain, res_idx) → original auth_seq_id
        #
        # Then for each output chain (sequential letter[i]):
        #   original chain = label_to_auth[ topology.chains()[i].id ]
        #
        # And residue numbers are restored by positional index within that auth chain.

        # source label → auth
        label_to_auth: dict[str, str] = {}
        # (source_auth_chain, heavy-atom res_idx) → original auth_seq_id
        seq_map: dict[tuple, str] = {}
        try:
            src_table = source_block.find(
                "_atom_site.",
                ["label_asym_id", "auth_asym_id", "auth_seq_id", "auth_atom_id"],
            )
            s_prev_auth = ""
            s_prev_seq  = ""
            s_res_idx   = -1
            for row in src_table:
                label, auth, seq, atom = row[0], row[1], row[2], row[3]
                if label not in label_to_auth:
                    label_to_auth[label] = auth
                if not str(atom).startswith("H"):
                    if auth != s_prev_auth:
                        s_prev_auth, s_prev_seq, s_res_idx = auth, seq, 0
                    elif seq != s_prev_seq:
                        s_prev_seq = seq
                        s_res_idx += 1
                    key = (auth, s_res_idx)
                    if key not in seq_map:
                        seq_map[key] = seq
        except Exception:
            pass

        # output sequential letter → original auth chain ID
        topo_chain_ids = [c.id for c in topology.chains()]
        seen_out_chains: list[str] = []
        try:
            for row in output_block.find("_atom_site.", ["auth_asym_id"]):
                ch = row[0]
                if ch not in seen_out_chains:
                    seen_out_chains.append(ch)
        except Exception:
            pass
        out_to_auth: dict[str, str] = {
            out_ch: label_to_auth.get(topo_ch, topo_ch)
            for out_ch, topo_ch in zip(seen_out_chains, topo_chain_ids)
        }

        # ── Patch the output CIF ──────────────────────────────────────────────────
        out_table = output_block.find(
            "_atom_site.",
            ["auth_asym_id", "auth_seq_id", "auth_atom_id", "label_asym_id"],
        )
        if out_table and out_to_auth:
            o_res_idx:  dict[str, int]   = {}
            o_prev_key: dict[str, tuple] = {}
            for row in out_table:
                out_ch, seq, atom = row[0], row[1], row[2]
                auth_ch = out_to_auth.get(out_ch, out_ch)
                res_key = (out_ch, seq)
                if auth_ch not in o_res_idx:
                    o_res_idx[auth_ch]  = 0
                    o_prev_key[auth_ch] = res_key
                elif res_key != o_prev_key[auth_ch]:
                    o_res_idx[auth_ch] += 1
                    o_prev_key[auth_ch] = res_key
                row[0] = auth_ch   # auth_asym_id  → original auth
                row[3] = auth_ch   # label_asym_id → same, for consistency
                orig_seq = seq_map.get((auth_ch, o_res_idx[auth_ch]))
                if orig_seq is not None:
                    row[1] = orig_seq  # auth_seq_id → original residue number

        output_doc.write_file(str(output_path))
        _patch_nonstd_bonds_in_cif(output_path, topology)
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
