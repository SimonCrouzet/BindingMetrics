"""Structure loading and manipulation utilities."""

import functools
import logging
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from binding_metrics.core.residues import (
    PHOSPHO_RESIDUES,
    PROTEIN_RESIDUES,
    WATER_NAMES_STRIP_HETEROGENS,
)
from binding_metrics.utils import add_to_report, backfill_auth_columns, extend_report

if TYPE_CHECKING:
    from openmm.app import PDBFile

logger = logging.getLogger(__name__)


def __getattr__(name: str):
    """Resolve ``app`` and ``PDBFile``, which this module imported at load time (PEP 562).

    OpenMM is imported inside the functions that need it, so that chain detection
    and CIF writing work on an install without the ``simulation`` extra.
    """
    if name == "app":
        from openmm import app

        return app
    if name == "PDBFile":
        from openmm.app import PDBFile

        return PDBFile
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_complex(pdb_path: str | Path) -> "PDBFile":
    """Load a PDB file containing a molecular complex.

    Args:
        pdb_path: Path to the PDB file

    Returns:
        Loaded PDBFile object

    Raises:
        FileNotFoundError: If the PDB file doesn't exist
        ValueError: If the file cannot be parsed
    """
    from openmm.app import PDBFile

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
    from openmm.app import PDBFile

    pdb = PDBFile(str(pdb_path))
    topology = pdb.topology

    indices = []
    for atom in topology.atoms():
        if atom.residue.chain.id in chain_ids:
            indices.append(atom.index)

    return indices


#: ``_struct_conn.conn_type_id`` values that OpenMM's PDBxFile turns into bonds.
_STRUCT_CONN_BOND_TYPES = ("covale", "disulf", "modres")


def _restore_struct_conn_bonds(path: Path, topology) -> int:
    """Add the ``_struct_conn`` bonds that ``PDBxFile`` failed to resolve.

    ``PDBxFile`` keys every atom by ``auth_seq_id`` (and, in most files, by
    ``auth_asym_id``) but looks the two partners of a ``_struct_conn`` row up by
    ``ptnr*_label_seq_id`` and ``ptnr*_label_asym_id``. The numberings differ
    whenever a chain does not start at residue 1 (the 1QJB peptide has label
    numbers 4 and 5 for author numbers 6 and 7), so the row matches nothing and
    the bond is dropped without a message. That is fatal for a link into a
    non-standard residue such as phosphoserine: ``createStandardBonds`` only
    knows the standard residue types, so the HIS-SEP peptide bond exists nowhere
    else and the force field later reports "bonds are different".

    Partners are matched through ``_atom_site.id``, which OpenMM keeps as
    ``Atom.id``. The author columns of ``_struct_conn`` are used when present
    (wwPDB files always have them) and the label columns otherwise. A partner
    that matches no atom, or more than one, is skipped.

    Returns the number of bonds added.
    """
    try:
        import gemmi
    except ImportError:
        logger.info(
            "gemmi is not installed: covalent links of %s whose label and author residue "
            "numbers differ are not restored. Install with: pip install binding-metrics[structure]",
            path.name,
        )
        return 0
    try:
        block = gemmi.cif.read(str(path))[0]
    except (RuntimeError, ValueError, IndexError):
        return 0  # PDBxFile has already parsed the file; nothing more to add here

    for scheme in ("auth", "label"):
        # Only the author numbering can carry an insertion code.
        conn_ins_cols = ["?pdbx_ptnr1_PDB_ins_code", "?pdbx_ptnr2_PDB_ins_code"]
        site_ins_cols = ["?pdbx_PDB_ins_code"]
        if scheme == "label":
            conn_ins_cols = site_ins_cols = []
        partner_cols = [f"ptnr{n}_{scheme}_{col}" for n in "12" for col in ("asym_id", "seq_id")]
        conn = block.find(
            "_struct_conn.",
            ["conn_type_id", *partner_cols, "ptnr1_label_atom_id", "ptnr2_label_atom_id"]
            + conn_ins_cols,
        )
        site = block.find(
            "_atom_site.",
            ["id", f"{scheme}_asym_id", f"{scheme}_seq_id", "label_atom_id"] + site_ins_cols,
        )
        if conn and site:
            break
    else:
        return 0

    def _ins_code(table, row, col: int) -> str:
        """Insertion code, or "" when the column is absent or null."""
        if col >= len(row) or not table.has_column(col) or row.str(col) in ("?", "."):
            return ""
        return row.str(col)

    atom_by_id = {str(atom.id): atom for atom in topology.atoms()}
    atoms_by_key: dict[tuple, list] = {}
    for row in site:
        atom = atom_by_id.get(row.str(0))
        if atom is not None:  # alternate locations and later models have no Atom
            key = (row.str(1), row.str(2), _ins_code(site, row, 4), row.str(3))
            atoms_by_key.setdefault(key, []).append(atom)

    existing = {frozenset((b.atom1.index, b.atom2.index)) for b in topology.bonds()}
    added = 0
    for row in conn:
        if row.str(0)[:6] not in _STRUCT_CONN_BOND_TYPES:
            continue
        partners = []
        for n, ins_col in ((0, 7), (1, 8)):
            key = (
                row.str(1 + 2 * n),
                row.str(2 + 2 * n),
                _ins_code(conn, row, ins_col),
                row.str(5 + n),
            )
            partners.append(atoms_by_key.get(key, []))
        if len(partners[0]) != 1 or len(partners[1]) != 1:
            continue  # absent, or ambiguous (label numbering of a branched entity)
        atom1, atom2 = partners[0][0], partners[1][0]
        pair = frozenset((atom1.index, atom2.index))
        if atom1.index != atom2.index and pair not in existing:
            topology.addBond(atom1, atom2)
            existing.add(pair)
            added += 1
    return added


def _bond_bare_residues(topology, positions) -> int:
    """Bond, by covalent radii, every multi-atom residue that has no bond at all.

    A raw mmCIF lists no bond inside a non-standard residue (the wwPDB leaves them
    to the Chemical Component Dictionary), and ``createStandardBonds`` knows only
    the standard residue types, so phosphoserine and the like load as a cloud of
    unbonded atoms. A PDB file carries them as CONECT records. Without the bonds
    the force field cannot match the residue ("bonds are different"). The peptide
    chain gets this repair again in ``patch_cyclic_topology``; no other chain does.

    Returns the number of bonds added.
    """
    from binding_metrics.core.cyclic import reconstruct_intraresidue_bonds

    bonded = {b.atom1.residue.index for b in topology.bonds() if b.atom1.residue is b.atom2.residue}
    added = 0
    for chain in topology.chains():
        residues = list(chain.residues())
        if any(res.index not in bonded and sum(1 for _ in res.atoms()) > 1 for res in residues):
            added += reconstruct_intraresidue_bonds(
                topology, positions, chain.id, residues=residues
            )
    return added


def load_structure(path: str | Path) -> tuple:
    """Load a structure file (PDB or CIF) and return (topology, positions).

    Supports .pdb, .cif, and .mmcif formats. For a CIF, the ``covale``, ``disulf``
    and ``modres`` rows of ``_struct_conn`` are bonded even when the label and
    author numbering of the file differ (with gemmi installed), and a residue
    that the file gives no bond at all (phosphoserine, an NCAA) is bonded by
    covalent radii, as a PDB file's CONECT records would do.

    Args:
        path: Path to the structure file

    Returns:
        Tuple of (topology, positions) as OpenMM objects

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the format is unsupported or parsing fails
    """
    from openmm.app import PDBFile, PDBxFile

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix in (".cif", ".mmcif"):
            struct = PDBxFile(str(path))
            n_restored = _restore_struct_conn_bonds(path, struct.topology)
            if n_restored:
                logger.debug("%s: restored %d _struct_conn bond(s)", path.name, n_restored)
            _bond_bare_residues(struct.topology, struct.positions)
            attach_author_chain_ids(struct.topology, path)
        elif suffix == ".pdb":
            struct = PDBFile(str(path))
        else:
            raise ValueError(f"Unsupported structure format: {suffix}. Use .pdb, .cif, or .mmcif")
    except (ValueError, FileNotFoundError):
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse structure file {path}: {e}") from e

    return struct.topology, struct.positions


#: Attribute that carries the author chain IDs on an OpenMM ``Topology``: a tuple with
#: one ``(OpenMM chain ID, author chain ID)`` pair per chain, in chain order.
_AUTHOR_IDS_ATTRIBUTE = "_binding_metrics_author_chain_ids"


def attach_author_chain_ids(topology, source_path: str | Path) -> bool:
    """Record on ``topology`` the author chain ID (``auth_asym_id``) of each of its chains.

    OpenMM's ``PDBxFile`` names the chains of an mmCIF by ``label_asym_id`` when the
    file has strictly more label IDs than author IDs (every water and ligand asym
    unit has a label ID of its own), and by the author ID otherwise. The chain
    options of the package, the metrics that read the file with biotite and the
    results use the author ID, so the peptide of 1CWA is chain C for them and chain
    B in the OpenMM topology. The topology keeps its own IDs, because the waters
    of a chain share its author ID and renaming would merge them with the protein.
    This function stores the author ID beside them; :func:`author_chain_ids` reads
    it and :func:`openmm_chain_id` turns an author ID into the chain to look up.

    Atoms are matched by ``Atom.id``, the ``_atom_site.id`` of the row an atom was read
    from, which ``Modeller`` keeps, and must agree with the row on the element. A topology
    that has lost atoms or chains since the file was read can still be annotated from it;
    one that has gained atoms cannot, because ``Topology`` gives a new atom an ID of its
    own that may be a row's. A PDB file needs no call: its chain IDs are the author IDs.

    Args:
        topology: OpenMM Topology read from ``source_path``.
        source_path: The mmCIF file.

    Returns:
        True when the author IDs were recorded. False, with a log record, when gemmi
        is not installed, the file cannot be read, or the atoms of the topology are not
        those of the file; the author IDs are then taken to be the topology's own.
    """
    path = Path(source_path)
    try:
        import gemmi
    except ImportError:
        logger.info(
            "gemmi is not installed: the chains of %s keep OpenMM's chain IDs, which are the "
            "label IDs when the file has more label IDs than author IDs. Install with: "
            "pip install binding-metrics[structure]",
            path.name,
        )
        return False
    try:
        block = gemmi.cif.read(str(path))[0]
        table = block.find("_atom_site.", ["id", "label_asym_id", "?auth_asym_id", "?type_symbol"])
    except (RuntimeError, ValueError, IndexError) as exc:
        logger.warning(
            "cannot read the author chain IDs of %s (%s: %s)", path.name, type(exc).__name__, exc
        )
        return False
    if not table:
        return False
    author_column = 2 if table.has_column(2) else 1  # PDBxFile falls back to label_asym_id too
    site_of_atom = {
        row.str(0): (row.str(author_column), row.str(3).upper() if table.has_column(3) else "")
        for row in table
    }

    recorded = []
    n_matched = 0
    for chain in topology.chains():
        authors = set()
        for atom in chain.atoms():
            site = site_of_atom.get(str(atom.id))
            if site is None:
                continue
            symbol = atom.element.symbol.upper() if atom.element is not None else ""
            if site[1] and symbol and site[1] != symbol:
                # The same ID on another element: the atom is not one of this file's.
                logger.warning(
                    "atom %s of chain %s is %s in the topology and %s in %s: the topology was "
                    "not read from this file, the author chain IDs are not recorded",
                    atom.id,
                    chain.id,
                    symbol,
                    site[1],
                    path.name,
                )
                return False
            authors.add(site[0])
        if len(authors) > 1:
            logger.warning(
                "chain %s of the topology holds atoms of the author chains %s in %s: the author "
                "IDs are not recorded",
                chain.id,
                ", ".join(sorted(authors)),
                path.name,
            )
            return False
        n_matched += bool(authors)
        recorded.append((chain.id, authors.pop() if authors else chain.id))
    if not n_matched:
        return False
    setattr(topology, _AUTHOR_IDS_ATTRIBUTE, tuple(recorded))
    return True


def _recorded_author_ids(topology) -> Optional[list[str]]:
    """The author IDs that :func:`attach_author_chain_ids` stored, or None when none apply.

    A record that no longer fits (another number of chains, or a chain that was renamed)
    belongs to an earlier state of the topology and is ignored.
    """
    recorded = getattr(topology, _AUTHOR_IDS_ATTRIBUTE, None)
    chains = list(topology.chains())
    if recorded is None or len(recorded) != len(chains):
        return None
    if any(openmm_id != chain.id for (openmm_id, _), chain in zip(recorded, chains)):
        return None
    return [author_id for _, author_id in recorded]


def author_chain_ids(topology) -> list[str]:
    """Author chain ID of every chain of an OpenMM topology, in chain order.

    The IDs come from :func:`attach_author_chain_ids` (``load_structure`` calls it for a
    CIF). A topology without them, such as one read from a PDB file, gives the
    chain's own ID, which is the author ID there.
    """
    recorded = _recorded_author_ids(topology)
    return recorded if recorded is not None else [chain.id for chain in topology.chains()]


def copy_author_chain_ids(source, target) -> None:
    """Give ``target`` the author IDs of ``source``, matching the chains by ID in order.

    For a topology that was built from ``source`` with ``Modeller`` or PDBFixer and
    still carries the chain IDs of ``source``, some chains possibly removed. A chain of
    ``target`` that matches none keeps its own ID as its author ID. Nothing is done when
    ``source`` has no author IDs.
    """
    authors = _recorded_author_ids(source)
    if authors is None:
        return
    pairs = [(chain.id, author) for chain, author in zip(source.chains(), authors)]
    recorded = []
    cursor = 0
    for chain in target.chains():
        match = next((k for k in range(cursor, len(pairs)) if pairs[k][0] == chain.id), None)
        if match is None:
            recorded.append((chain.id, chain.id))
        else:
            recorded.append((chain.id, pairs[match][1]))
            cursor = match + 1
    setattr(target, _AUTHOR_IDS_ATTRIBUTE, tuple(recorded))


def openmm_chain_id(topology, author_id: str) -> Optional[str]:
    """ID, in ``topology``, of the amino-acid chain that the author chain ID names.

    A chain counts when it holds at least one amino-acid residue (the set of
    :func:`detect_chains`). The waters and ligands that OpenMM splits off an author
    chain are therefore never the answer: for 1CWA, author ID A is OpenMM chain A (the
    protein) and not C (its waters), and author ID C is chain B (the peptide) and not D.

    Args:
        topology: OpenMM Topology; the author IDs come from :func:`author_chain_ids`.
        author_id: The chain ID as the user gives it.

    Returns:
        The chain ID to look up in the topology, or None when no amino-acid chain has that
        author ID.

    Raises:
        ValueError: If amino-acid residues of that author ID sit in chains with
            different IDs in the topology, so that the chain is not determined.
    """
    amino_acids = _amino_acid_names()
    holders: list[str] = []
    for chain, author in zip(topology.chains(), author_chain_ids(topology)):
        if author != author_id or chain.id in holders:
            continue
        if any(residue.name in amino_acids for residue in chain.residues()):
            holders.append(chain.id)
    if len(holders) > 1:
        raise ValueError(
            f"author chain {author_id!r} holds amino-acid residues in the topology chains "
            f"{', '.join(map(repr, holders))}; name one of them by its topology ID"
        )
    return holders[0] if holders else None


def detect_chains(topology) -> tuple[Optional[str], Optional[str]]:
    """Auto-detect ligand (peptide) and receptor chain IDs from topology.

    Identifies protein chains by counting amino-acid residues: the standard ones and
    their AMBER variants, D-amino acids, phosphorylated residues and, with biotite
    installed, every peptide-linking component of the Chemical Component Dictionary
    (the definition of ``biotite.structure.filter_amino_acids``, which the interface
    metrics use). A chain of D-residues is a protein chain.
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
    amino_acids = _amino_acid_names()
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
        verbose: If True, log the detected chains at INFO level.

    Returns:
        Dict with keys:
            peptide_chain (str): resolved peptide chain ID (author ID,
                ``auth_asym_id`` in a CIF)
            receptor_chain (str): resolved receptor chain ID (author ID); None
                when the file has a single protein chain
            peptide_chain_label (str): the peptide chain ID as OpenMM sees it, which
                OpenMM-based steps need. OpenMM's PDBxFile names the chains by
                ``label_asym_id`` when the file has more distinct label IDs than
                author IDs (waters and ligands each get their own label ID), and
                by the author ID otherwise. It equals ``peptide_chain`` for PDB
                files, for CIFs whose author and label IDs agree, for CIFs with
                as many author IDs as label IDs, and when the mapping cannot be
                built (a warning is logged then).
            receptor_chain_label (str): same for the receptor chain
            peptide_n_residues (int): number of residues in peptide chain
            receptor_n_residues (int): number of residues in receptor chain
            all_chains (list[dict]): all protein chains with id and n_residues

        A chain ID passed in ``peptide_chain`` or ``receptor_chain`` is returned
        as given, even when it is not in the file; its residue count is then None.
    """
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure.io.pdbx as pdbx

    path = Path(path)
    suffix = path.suffix.lower()

    # Build auth→label chain ID mapping for CIF files.
    # biotite uses auth_asym_id by default; OpenMM uses label_asym_id when the file
    # has more label IDs than author IDs, and auth_asym_id otherwise. OpenMM-based
    # steps need the ID OpenMM used.
    # We restrict the mapping to CA atoms so that ligand/water label chains
    # (which share the same auth chain as the protein) don't overwrite the
    # protein label.
    auth_to_label: dict[str, str] = {}
    if suffix in (".cif", ".mmcif"):
        f = pdbx.CIFFile.read(str(path))
        backfill_auth_columns(f)
        atoms = pdbx.get_structure(f, model=1)
        try:
            atom_site = f.block["atom_site"]
            auth_ids = atom_site["auth_asym_id"].as_array()
            label_ids = atom_site["label_asym_id"].as_array()
            atom_names = atom_site["auth_atom_id"].as_array()
            for auth, label, name in zip(auth_ids, label_ids, atom_names):
                if str(name).strip() == "CA":  # protein Cα only
                    a_str, l_str = str(auth), str(label)
                    if a_str not in auth_to_label:  # first occurrence wins
                        auth_to_label[a_str] = l_str
            if len(set(label_ids)) <= len(set(auth_ids)):
                # OpenMM's PDBxFile keeps the author IDs unless there are strictly
                # more label IDs; a label ID would then name another chain.
                auth_to_label = {}
        except (KeyError, ValueError) as exc:
            # Without the label/auth pair the mapping stays empty, and the label
            # chain IDs handed to OpenMM-based steps fall back to the author IDs.
            logger.warning(
                "%s: cannot map author chain IDs to label chain IDs (%s: %s); "
                "assuming they are identical",
                path.name,
                type(exc).__name__,
                exc,
            )
    else:
        f = pdb_io.PDBFile.read(str(path))
        atoms = pdb_io.get_structure(f, model=1)

    # Count standard amino-acid residues per chain
    aa_filter = struc.filter_amino_acids(atoms)
    aa_atoms = atoms[aa_filter]
    chain_ids = sorted(set(aa_atoms.chain_id))

    chain_info = []
    for cid in chain_ids:
        n_res = len(
            set(
                zip(
                    aa_atoms.chain_id[aa_atoms.chain_id == cid],
                    aa_atoms.res_id[aa_atoms.chain_id == cid],
                )
            )
        )
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
            pep_ca = aa_atoms[(aa_atoms.chain_id == peptide_chain) & (aa_atoms.atom_name == "CA")]
            best_chain, best_contacts = None, -1
            for ci in chain_info:
                if ci["id"] == peptide_chain:
                    continue
                cand_ca = aa_atoms[(aa_atoms.chain_id == ci["id"]) & (aa_atoms.atom_name == "CA")]
                if len(pep_ca) == 0 or len(cand_ca) == 0:
                    contacts = 0
                else:
                    import numpy as np
                    from biotite.structure import distance

                    dists = np.array(
                        [
                            distance(pep_ca.coord, cand_ca.coord[j]).min()
                            for j in range(len(cand_ca))
                        ]
                    )
                    contacts = int((dists < 8.0).sum())
                if contacts > best_contacts:
                    best_contacts, best_chain = contacts, ci["id"]
            receptor_chain = best_chain
        rec_auto = True
    else:
        rec_auto = receptor_chain is None

    pep_info = next((c for c in chain_info if c["id"] == peptide_chain), None)
    rec_info = (
        next((c for c in chain_info if c["id"] == receptor_chain), None) if receptor_chain else None
    )

    if verbose:
        logger.info("  Chain detection (%s):", path.name)
        for c in chain_info:
            tag = ""
            if c["id"] == peptide_chain:
                tag = "  ← peptide" + (" [auto]" if pep_auto else "")
            elif c["id"] == receptor_chain:
                tag = "  ← receptor" + (" [auto]" if rec_auto else "")
            logger.info("    chain %s: %s residues%s", c["id"], c["n_residues"], tag)

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
    report: Optional[dict] = None,
):
    """Remove non-protein residues from topology, warning if close to the interface.

    Amino-acid residues and the phosphorylated residues SEP, TPO and PTR count as
    protein and are kept in every chain, so an unselected chain is not cut where it
    carries a phosphoserine.

    Args:
        topology: OpenMM Topology (post-PDBFixer).
        positions: Atom positions (OpenMM Quantity, nm).
        peptide_chain: Peptide chain ID to preserve.
        receptor_chain: Receptor chain ID to preserve.
        warn_cutoff_ang: Distance threshold in Å; heterogens within this distance
            trigger a warning before removal.
        report: Optional dict filled in place with what was removed. Lists and
            counts accumulate when one dict is passed to several calls. Keys:

            * ``removed_heterogens`` (list[str]): ``"NAME (chain X)"`` for each
              removed non-water heterogen (ligands, ions, glycans). X is the author
              chain ID (:func:`author_chain_ids`), also in the log lines.
            * ``n_removed_waters`` (int): water molecules removed.

            Behaviour is identical when ``report`` is None.

    Returns:
        Tuple (topology, positions) with heterogens removed. The topology keeps the
        author chain IDs of the one passed in.
    """
    import numpy as np

    protein_chain_ids = {c for c in (peptide_chain, receptor_chain) if c}
    author_ids = author_chain_ids(topology)  # for the messages; the selection uses chain.id
    protein_pos = (
        np.array(
            [
                [p.x, p.y, p.z]
                for a, p in zip(topology.atoms(), positions)
                if a.residue.chain.id in protein_chain_ids
            ]
        )
        * 10
    )  # nm → Å

    atoms_to_remove = []
    removed_heterogens: list[str] = []
    n_removed_waters = 0
    for res in topology.residues():
        if res.chain.id in protein_chain_ids:
            continue
        if res.name in PROTEIN_RESIDUES or res.name in PHOSPHO_RESIDUES:
            continue  # peptide-linked: deleting it would cut the chain in two
        # Water: always remove silently
        if res.name in WATER_NAMES_STRIP_HETEROGENS:
            atoms_to_remove.extend(res.atoms())
            n_removed_waters += 1
            continue
        author_id = author_ids[res.chain.index]
        removed_heterogens.append(f"{res.name} (chain {author_id})")
        # Other heterogens: warn if close to protein (may be a cofactor/ion)
        res_pos = (
            np.array(
                [
                    [positions[a.index].x, positions[a.index].y, positions[a.index].z]
                    for a in res.atoms()
                ]
            )
            * 10
        )  # nm → Å
        if len(protein_pos) > 0 and len(res_pos) > 0:
            dists = np.linalg.norm(res_pos[:, None, :] - protein_pos[None, :, :], axis=-1)
            min_dist = float(dists.min())
            if min_dist < warn_cutoff_ang:
                logger.warning(
                    "  Warning: removing heterogen %s%s "
                    "(chain %s) which is %.1f Å from "
                    "the protein — it may be a functional cofactor or ion. "
                    "Parametrize it via custom_bond_handler to keep it.",
                    res.name,
                    res.id,
                    author_id,
                    min_dist,
                )
            else:
                logger.info(
                    "  Removing distant heterogen %s%s (chain %s, %.1f Å from protein)",
                    res.name,
                    res.id,
                    author_id,
                    min_dist,
                )
        else:
            logger.info("  Removing heterogen %s%s (chain %s)", res.name, res.id, author_id)
        atoms_to_remove.extend(res.atoms())

    if report is not None:
        extend_report(report, "removed_heterogens", removed_heterogens)
        add_to_report(report, "n_removed_waters", n_removed_waters)

    if atoms_to_remove:
        from openmm import app

        modeller = app.Modeller(topology, positions)
        modeller.delete(atoms_to_remove)
        copy_author_chain_ids(topology, modeller.topology)
        topology, positions = modeller.topology, modeller.positions

    return topology, positions


@functools.lru_cache(maxsize=1)
def _amino_acid_names() -> frozenset:
    """Residue names that count as amino acids when a chain is called a protein chain.

    The package's own protein set (standard residues, AMBER variants, the lactam and
    N-methyl templates), the phosphorylated residues, the D-amino acids of
    ``core.nonstandard.D_AA_MAP`` and, with biotite installed, every peptide-linking
    component of the Chemical Component Dictionary (the set behind
    ``biotite.structure.filter_amino_acids``, which the interface metrics use).
    """
    from binding_metrics.core.nonstandard import D_AA_MAP

    names = set(PROTEIN_RESIDUES) | set(PHOSPHO_RESIDUES) | set(D_AA_MAP)
    try:
        from biotite.structure.info import amino_acid_names
    except ImportError:
        pass  # the sets above still cover the residues the package parameterises
    else:
        names |= set(amino_acid_names())
    return frozenset(names)


def drop_other_protein_chains(
    topology,
    positions,
    peptide_chain: Optional[str],
    receptor_chain: Optional[str],
    report: Optional[dict] = None,
):
    """Delete every protein chain that is neither the peptide nor the receptor.

    The relaxation and the interaction energy describe the peptide-receptor pair. A
    third protein chain (a second copy of the complex in the asymmetric unit, a
    crystallisation partner) adds its own energy to the complex but not to the
    isolated components, so E_int would mix the pair with the bystander, and its caps,
    phosphorylated residues and lactam bridges are not patched, so the force field
    often cannot build it. Call after :func:`strip_heterogens`. Nothing is removed
    unless both chain IDs are given and both are in the topology. A logged warning
    names each removed chain: a receptor made of several chains (a Fab) has to be
    reduced to the chain that carries the interface before it goes in.

    Args:
        topology: OpenMM Topology.
        positions: Atom positions (OpenMM Quantity, nm).
        peptide_chain: Peptide chain ID to keep.
        receptor_chain: Receptor chain ID to keep.
        report: Optional dict; ``dropped_protein_chains`` (list[str]) receives the
            IDs of the removed chains, an empty list when none was removed. They are
            author chain IDs (:func:`author_chain_ids`), as in the warning. Lists
            accumulate when one dict is passed to several calls.

    Returns:
        Tuple (topology, positions) without the other protein chains, with the author
        chain IDs of the topology passed in.
    """
    kept = {peptide_chain, receptor_chain}
    present = {chain.id for chain in topology.chains()}
    if not (peptide_chain and receptor_chain) or not kept <= present:
        if report is not None:
            extend_report(report, "dropped_protein_chains", [])
        return topology, positions

    amino_acids = _amino_acid_names()
    author_ids = author_chain_ids(topology)  # for the messages; the selection uses chain.id
    atoms_to_remove = []
    dropped: list[str] = []
    for chain in topology.chains():
        if chain.id in kept or not any(res.name in amino_acids for res in chain.residues()):
            continue
        dropped.append(author_ids[chain.index])
        atoms_to_remove.extend(atom for res in chain.residues() for atom in res.atoms())

    if report is not None:
        extend_report(report, "dropped_protein_chains", dropped)
    if not dropped:
        return topology, positions

    author_of = {}
    for chain, author_id in zip(topology.chains(), author_ids):
        author_of.setdefault(chain.id, author_id)
    logger.warning(
        "  Removing protein chain(s) %s: neither the peptide (%s) nor the receptor (%s). "
        "The energy and the relaxation describe the peptide-receptor pair only.",
        ", ".join(dropped),
        author_of[peptide_chain],
        author_of[receptor_chain],
    )
    from openmm import app

    modeller = app.Modeller(topology, positions)
    modeller.delete(atoms_to_remove)
    copy_author_chain_ids(topology, modeller.topology)
    return modeller.topology, modeller.positions


def _patch_nonstd_bonds_in_cif(cif_path: Path, topology) -> None:
    """Add non-disulfide non-sequential intra-chain bonds to _struct_conn.

    PDBxFile.writeFile only records disulfide bonds.  Custom covalent bonds
    (head-to-tail amide, lactam, etc.) are silently omitted, so PDBxFile
    cannot round-trip them.  This function reads the written CIF back with
    gemmi and appends the missing covale rows.

    Without gemmi the bonds cannot be written, so a saved cyclic peptide comes
    back linear on reload; that is logged as a warning when such bonds exist.
    """
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

    try:
        import gemmi
    except ImportError:
        logger.warning(
            "gemmi is not installed: the %d ring-closure or other non-sequential "
            "bond(s) of %s cannot be written to _struct_conn, so the file reloads "
            "without them (a cyclic peptide comes back linear). Install with: "
            "pip install binding-metrics[structure]",
            len(custom_bonds),
            cif_path,
        )
        return

    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()

    # Derive each residue's (auth_asym_id, auth_seq_id) from the FINAL _atom_site
    # that was just written.  Those rows are 1:1 with topology.atoms(), and those
    # are the exact columns OpenMM's PDBxFile keys atoms on when it reloads the
    # file.  Using the topology chain id / a chain-local index here instead would
    # not match _atom_site after save_cif has rewritten it to the original auth
    # IDs, and OpenMM would silently drop the bond on reload.
    res_key: dict[int, tuple[str, str]] = {}
    site = block.find("_atom_site.", ["auth_asym_id", "auth_seq_id"])
    if site:
        for atom, row in zip(topology.atoms(), site):
            res_key.setdefault(atom.residue.index, (row[0], row[1]))

    def _asym_seq(res) -> tuple[str, str]:
        # Fall back to the topology's own values if _atom_site could not be read.
        return res_key.get(res.index, (res.chain.id, str(res.id)))

    existing = block.find(["_struct_conn.id"])
    n_existing = len(existing) if existing else 0

    # Use the SAME label_* column names PDBxFile.writeFile emits.  On reload
    # OpenMM reads struct_conn partners exclusively via ptnr*_label_asym_id /
    # ptnr*_label_seq_id / ptnr*_label_atom_id — a freshly-created loop that used
    # auth_* column names would be written but never read back.  The values we
    # store come from _atom_site's auth_asym_id / auth_seq_id (which save_cif has
    # already made equal to the label_asym_id it writes), so they match the atom
    # keys OpenMM builds.
    _STRUCT_CONN_COLS = [
        "_struct_conn.id",
        "_struct_conn.conn_type_id",
        "_struct_conn.ptnr1_label_asym_id",
        "_struct_conn.ptnr1_label_comp_id",
        "_struct_conn.ptnr1_label_seq_id",
        "_struct_conn.ptnr1_label_atom_id",
        "_struct_conn.ptnr2_label_asym_id",
        "_struct_conn.ptnr2_label_comp_id",
        "_struct_conn.ptnr2_label_seq_id",
        "_struct_conn.ptnr2_label_atom_id",
    ]
    loop_ref = block.find_loop("_struct_conn.id")
    if loop_ref:
        loop = loop_ref.get_loop()
    else:
        loop = block.init_loop("_struct_conn.", [col.split(".")[1] for col in _STRUCT_CONN_COLS])

    for i, (a1, a2) in enumerate(custom_bonds):
        r1, r2 = a1.residue, a2.residue
        asym1, seq1 = _asym_seq(r1)
        asym2, seq2 = _asym_seq(r2)
        bond_id = f"covale{n_existing + i + 1}"
        loop.add_row(
            [
                bond_id,
                "covale",
                asym1,
                r1.name,
                seq1,
                a1.name,
                asym2,
                r2.name,
                seq2,
                a2.name,
            ]
        )

    doc.write_file(str(cif_path))


#: Force-field-internal residue names that prep introduces to match specialised
#: templates, mapped back to the standard code they must be restored to on
#: output. CYX = disulfide-bonded CYS; ASPL/GLUL/LYSL = the lactam-closing
#: Asp/Glu/Lys templates.
_FF_INTERNAL_RESIDUE_RENAMES = (
    ("CYX", "CYS"),
    ("ASPL", "ASP"),
    ("GLUL", "GLU"),
    ("LYSL", "LYS"),
)


def _rename_internal_residues_to_standard(cif_path: Path) -> None:
    """Restore force-field-internal residue names to standard codes on output.

    Prep renames residues internally so they match specialised force-field
    templates — disulfide CYS→CYX, and lactam-closing ASP/GLU/LYS→ASPL/GLUL/LYSL.
    Those internal names must not leak into output files: downstream tools, and
    binding-metrics' own ``detect_cyclization`` (which matches the standard
    residue name plus the ``_struct_conn`` closure bond), expect CYS/ASP/GLU/LYS.
    The bond geometry and ``_struct_conn`` records are untouched, so re-detection
    recovers the closure from the bond on reload. Without this, a saved lactam
    structure carries GLUL/LYSL names and fails re-detection, raising a spurious
    CyclizationError when the prepped file is relaxed.

    Uses whole-word text replacement rather than gemmi mutation because gemmi
    Table views from ``block.find()`` do not propagate edits back through
    ``doc.write_file()``. Each code is a 3–4-letter token unique to the
    force-field internals and safe to replace as a whole word.
    """
    import re

    content = cif_path.read_text(encoding="utf-8")
    new_content = content
    for internal, standard in _FF_INTERNAL_RESIDUE_RENAMES:
        if internal in new_content:
            new_content = re.sub(rf"\b{internal}\b", standard, new_content)
    if new_content != content:
        cif_path.write_text(new_content, encoding="utf-8")


def _ids_fit_cif(topology) -> bool:
    """True when every chain ID and residue number can be written to mmCIF as they are.

    ``PDBxFile.writeFile(keepIds=True)`` prints them unquoted, so a blank or
    spaced chain ID (a PDB file without chain IDs) or a non-numeric residue
    number would corrupt the columns. Chain IDs must also be unique: a PDB file
    puts the waters of chain A in a second chain called A, and once the two share
    an ID the metric code cannot tell them apart.
    """
    chain_ids = [chain.id for chain in topology.chains()]
    return (
        len(set(chain_ids)) == len(chain_ids)
        and all(re.fullmatch(r"[A-Za-z0-9]+", cid) for cid in chain_ids)
        and all(re.fullmatch(r"-?[0-9]+", str(res.id)) for res in topology.residues())
    )


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

    Falls back to raw OpenMM output if gemmi is not available (logged as a
    warning) or source is None.

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
        # Without a source CIF to read the caller's IDs from, the topology's own
        # chain IDs and residue numbers are the only ones there are. PDBxFile
        # would replace them with A, B, C... and 1, 2, 3...
        with open(output_path, "w", encoding="utf-8") as f:
            PDBxFile.writeFile(topology, positions, f, keepIds=_ids_fit_cif(topology))
        # PDBxFile only writes disulfide bonds to _struct_conn; patch in any
        # other non-sequential intra-chain covalent bonds (e.g. head-to-tail).
        _patch_nonstd_bonds_in_cif(output_path, topology)
        _rename_internal_residues_to_standard(output_path)
        return

    try:
        import gemmi
    except ImportError:
        logger.warning(
            "gemmi is not installed: %s keeps OpenMM's sequential chain IDs and "
            "1-based residue numbers instead of those of %s, and non-sequential "
            "bonds are not recorded. Install with: pip install binding-metrics[structure]",
            output_path,
            source_cif_path,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            PDBxFile.writeFile(topology, positions, f)
        _rename_internal_residues_to_standard(output_path)
        return

    # Write fresh OpenMM CIF — correct atoms and H, but with label chain IDs
    # and 1-based sequential auth_seq_id.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cif", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)
        PDBxFile.writeFile(topology, positions, tmp)

    try:
        source_block = gemmi.cif.read(str(source_cif_path)).sole_block()
        output_doc = gemmi.cif.read(str(tmp_path))
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
        #
        # PDBxFile names the chains by label_asym_id only when the file has more
        # label IDs than author IDs (waters and ligands each get a label ID);
        # otherwise the topology already carries the author IDs and the map stays
        # empty: a label ID could name another chain's author ID.

        # source label → auth
        label_to_auth: dict[str, str] = {}
        source_auth_ids: set[str] = set()
        # (source_auth_chain, heavy-atom res_idx) → original auth_seq_id
        seq_map: dict[tuple, str] = {}
        source_columns = ["label_asym_id", "auth_asym_id", "auth_seq_id", "auth_atom_id"]
        absent = [c for c in source_columns if not source_block.find_loop(f"_atom_site.{c}")]
        if absent:
            logger.warning(
                "%s lacks _atom_site column(s) %s: chain IDs and residue numbers of %s "
                "cannot be restored from it",
                Path(source_cif_path).name,
                ", ".join(absent),
                output_path.name,
            )
        try:
            src_table = source_block.find("_atom_site.", source_columns)
            s_prev_auth = ""
            s_prev_seq = ""
            s_res_idx = -1
            for row in src_table:
                label, auth, seq, atom = row[0], row[1], row[2], row[3]
                source_auth_ids.add(auth)
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
        except (RuntimeError, IndexError, ValueError) as exc:
            logger.warning(
                "cannot read the source residue numbering of %s (%s: %s); residue numbers "
                "of %s are not restored",
                Path(source_cif_path).name,
                type(exc).__name__,
                exc,
                output_path.name,
            )

        if len(label_to_auth) <= len(source_auth_ids):
            label_to_auth = {}

        # output sequential letter → original auth chain ID
        topo_chain_ids = [c.id for c in topology.chains()]
        seen_out_chains: list[str] = []
        try:
            for row in output_block.find("_atom_site.", ["auth_asym_id"]):
                ch = row[0]
                if ch not in seen_out_chains:
                    seen_out_chains.append(ch)
        except (RuntimeError, IndexError, ValueError) as exc:
            logger.warning(
                "cannot read the chain IDs OpenMM wrote to %s (%s: %s); original chain "
                "IDs are not restored",
                output_path.name,
                type(exc).__name__,
                exc,
            )
        out_to_auth: dict[str, str] = {
            out_ch: label_to_auth.get(topo_ch, topo_ch)
            for out_ch, topo_ch in zip(seen_out_chains, topo_chain_ids)
        }

        # ── Patch the output CIF ──────────────────────────────────────────────────
        out_table = output_block.find(
            "_atom_site.",
            ["auth_asym_id", "auth_seq_id", "auth_atom_id", "label_asym_id", "label_seq_id"],
        )
        # (chain, residue number) as OpenMM wrote them -> the residue number written
        # below. PDBxFile.writeFile puts the same 1-based number in label_seq_id and
        # in the _struct_conn partners; both must follow the auth_seq_id rewrite, or
        # OpenMM (which keys atoms by auth_seq_id and _struct_conn partners by
        # label_seq_id) cannot match a bond of a non-standard residue on reload.
        renumbered: dict[tuple, str] = {}
        if out_table and out_to_auth:
            o_res_idx: dict[str, int] = {}
            o_prev_key: dict[str, tuple] = {}
            # Residue numbers must stay unique per auth chain. Several output
            # chains can map back to one auth chain (waters especially), and
            # each of those carries its own 1-based numbering from PDBxFile —
            # so merging them without renumbering makes two distinct residues
            # share a number, and the reader folds them into one. That is a
            # silent structural mutation, so uniqueness wins over fidelity to
            # the source numbering where the two conflict.
            assigned: dict[tuple, str] = {}  # (auth_ch, res_idx) → seq to write
            used: dict[str, set] = {}  # auth_ch → seq values already taken
            for row in out_table:
                out_ch, seq, atom = row[0], row[1], row[2]
                auth_ch = out_to_auth.get(out_ch, out_ch)
                res_key = (out_ch, seq)
                if auth_ch not in o_res_idx:
                    o_res_idx[auth_ch] = 0
                    o_prev_key[auth_ch] = res_key
                elif res_key != o_prev_key[auth_ch]:
                    o_res_idx[auth_ch] += 1
                    o_prev_key[auth_ch] = res_key
                row[0] = auth_ch  # auth_asym_id  → original auth
                row[3] = auth_ch  # label_asym_id → same, for consistency

                res_id = (auth_ch, o_res_idx[auth_ch])
                if res_id not in assigned:
                    taken = used.setdefault(auth_ch, set())
                    candidate = seq_map.get(res_id, seq)
                    if candidate in taken:
                        # Number already belongs to a different residue: take
                        # the next free one above everything used so far.
                        numeric = [int(s) for s in taken if str(s).lstrip("-").isdigit()]
                        candidate = str(max(numeric) + 1 if numeric else 1)
                    taken.add(candidate)
                    assigned[res_id] = candidate
                row[1] = assigned[res_id]  # auth_seq_id → original, or unique
                row[4] = assigned[res_id]  # label_seq_id → same, see `renumbered`
                renumbered[(out_ch, seq)] = assigned[res_id]

        # Patch the _struct_conn residue numbers first: the chain patch below
        # replaces the OpenMM chain letters this lookup is keyed on.
        for chain_col, seq_col in (
            ("ptnr1_label_asym_id", "ptnr1_label_seq_id"),
            ("ptnr2_label_asym_id", "ptnr2_label_seq_id"),
        ):
            try:
                for row in output_block.find("_struct_conn.", [chain_col, seq_col]):
                    row[1] = renumbered.get((row[0], row[1]), row[1])
            except (RuntimeError, IndexError, ValueError) as exc:
                logger.warning(
                    "cannot restore residue numbers in _struct_conn.%s of %s (%s: %s)",
                    seq_col,
                    output_path.name,
                    type(exc).__name__,
                    exc,
                )

        # Patch _struct_conn chain IDs (PDBxFile writes label_asym_id with
        # sequential letters; auth_asym_id variants may also appear).
        if out_to_auth:
            for col in (
                "ptnr1_label_asym_id",
                "ptnr2_label_asym_id",
                "ptnr1_auth_asym_id",
                "ptnr2_auth_asym_id",
            ):
                try:
                    sc_table = output_block.find("_struct_conn.", [col])
                    for row in sc_table:
                        row[0] = out_to_auth.get(row[0], row[0])
                except (RuntimeError, IndexError, ValueError) as exc:
                    logger.warning(
                        "cannot restore original chain IDs in _struct_conn.%s of %s (%s: %s)",
                        col,
                        output_path.name,
                        type(exc).__name__,
                        exc,
                    )

        output_doc.write_file(str(output_path))
        _patch_nonstd_bonds_in_cif(output_path, topology)
        _rename_internal_residues_to_standard(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _append_missing_conect(topology, output_path: Path) -> None:
    """Add CONECT records for covalent bonds ``PDBFile.writeFile`` leaves out.

    OpenMM emits CONECT only when a bond touches a non-standard residue, or for
    a CYS SG-SG disulfide. A ring closure between two *standard* residues — the
    head-to-tail amide of a cyclic peptide, where the partners are an ordinary
    GLY and ASP — therefore leaves no trace in the file. On reload,
    ``createStandardBonds`` cannot infer it either, because it only bonds
    sequential residues and the two ends of a macrocycle are far apart in
    sequence. The peptide silently comes back linear.

    Only bonds OpenMM did not already write are appended, so no bond is
    declared twice.
    """
    from openmm.app import PDBFile

    written = set()
    for atom1, atom2 in topology.bonds():
        standard = PDBFile._standardResidues
        if atom1.residue.name not in standard or atom2.residue.name not in standard:
            written.add(frozenset((atom1.index, atom2.index)))
        elif (
            atom1.name == "SG"
            and atom2.name == "SG"
            and atom1.residue.name == "CYS"
            and atom2.residue.name == "CYS"
        ):
            written.add(frozenset((atom1.index, atom2.index)))

    missing = []
    for atom1, atom2 in topology.bonds():
        key = frozenset((atom1.index, atom2.index))
        if key in written:
            continue
        r1, r2 = atom1.residue, atom2.residue
        if r1.index == r2.index:
            continue  # intra-residue: rebuilt from the residue template
        if r1.chain.id == r2.chain.id and abs(r1.index - r2.index) == 1:
            continue  # sequential backbone: createStandardBonds handles it
        missing.append((atom1.index, atom2.index))

    if not missing:
        return

    lines = output_path.read_text(encoding="utf-8").splitlines(keepends=True)

    # Map topology atom order onto the serial numbers OpenMM actually wrote,
    # by reading them back rather than re-deriving its numbering (which skips
    # values for TER records).
    serials: list[str] = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            serials.append(line[6:11].strip())
    if len(serials) != topology.getNumAtoms():
        # Numbering could not be established; leaving the file as written is
        # better than emitting CONECT records pointing at the wrong atoms.
        return

    conect = [
        f"CONECT{int(serials[i]):>5}{int(serials[j]):>5}\n"
        for i, j in missing
        if serials[i].isdigit() and serials[j].isdigit()
    ]
    if not conect:
        return

    # CONECT must precede END / MASTER.
    insert_at = len(lines)
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].startswith(("END", "MASTER")):
            insert_at = idx
        elif lines[idx].strip():
            break
    output_path.write_text(
        "".join(lines[:insert_at] + conect + lines[insert_at:]), encoding="utf-8"
    )


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
        from openmm.app import PDBFile

        with open(output_path, "w", encoding="utf-8") as f:
            PDBFile.writeFile(topology, positions, f)
        _append_missing_conect(topology, output_path)
    else:
        raise ValueError(f"Unsupported output format: {suffix!r}. Use .pdb, .cif, or .mmcif")


def detect_models(path: str | Path) -> list[int]:
    """Read pdbx_PDB_model_num from a CIF and return sorted model numbers.

    Returns ``[1]`` for single-model CIFs (no ``pdbx_PDB_model_num`` column) and
    for PDB files. It also returns ``[1]``, with a logged warning, when the file
    cannot be read or biotite is not installed.

    Args:
        path: Path to the structure file.

    Returns:
        Sorted list of integer model numbers present in the file.
    """
    path = Path(path)
    if path.suffix.lower() not in (".cif", ".mmcif"):
        return [1]
    try:
        import biotite
        import biotite.structure.io.pdbx as pdbx
    except ImportError:
        logger.warning(
            "biotite is not installed: %s is treated as a single-model file. Install "
            "with: pip install binding-metrics[biotite]",
            path.name,
        )
        return [1]

    try:
        f = pdbx.CIFFile.read(str(path))
        atom_site = f.block["atom_site"]
        col = atom_site["pdbx_PDB_model_num"].as_array()
        unique = sorted({int(v) for v in col})
    except KeyError:
        return [1]  # no atom_site category or model column: a single-model file
    except (ValueError, OSError, biotite.InvalidFileError) as exc:
        logger.warning(
            "cannot read model numbers from %s (%s: %s); assuming a single model",
            path.name,
            type(exc).__name__,
            exc,
        )
        return [1]
    return unique if unique else [1]


def extract_model_to_tempfile(path: Path, model_num: int) -> Path:
    """Extract one model from a multi-model CIF into a temporary file.

    Uses gemmi to filter ``_atom_site`` rows where
    ``pdbx_PDB_model_num == model_num``.  All other CIF records (``_chem_comp``,
    ``_struct_conn``, etc.) are left intact so OpenMM can still parse the result.

    Returns the original *path* unchanged when:
    - The file is not a CIF.
    - The ``pdbx_PDB_model_num`` column is absent (already single-model).

    The caller is responsible for unlinking the returned temp file when it
    differs from the input path.

    Args:
        path: Path to the (possibly multi-model) CIF file.
        model_num: 1-based model number to extract.

    Returns:
        Path to a single-model temp CIF, or the original path if no extraction
        was needed.

    Raises:
        ValueError: If *model_num* is not found in the file.
        ImportError: If *path* is a CIF and gemmi is not installed. Returning
            the input would silently score whichever model the file lists first.
    """
    path = Path(path)
    if path.suffix.lower() not in (".cif", ".mmcif"):
        return path

    try:
        import gemmi
    except ImportError as exc:
        raise ImportError(
            f"gemmi is required to extract model {model_num} from {path.name}. "
            "Install with: pip install binding-metrics[structure]"
        ) from exc

    doc = gemmi.cif.read(str(path))
    block = doc.sole_block()

    loop_ref = block.find_loop("_atom_site.pdbx_PDB_model_num")
    if not loop_ref:
        return path  # single-model CIF

    loop = loop_ref.get_loop()
    n_cols = loop.width()
    n_rows = loop.length()
    tags = list(loop.tags)
    try:
        model_col = tags.index("_atom_site.pdbx_PDB_model_num")
    except ValueError:
        return path

    # loop.values is a flat row-major list; set_all_values() takes column-major.
    vals = list(loop.values)
    keep_rows = [i for i in range(n_rows) if vals[i * n_cols + model_col].strip() == str(model_num)]
    if not keep_rows:
        raise ValueError(f"Model {model_num} not found in {path}")

    columns = [[vals[r * n_cols + c] for r in keep_rows] for c in range(n_cols)]
    loop.set_all_values(columns)

    tmp = tempfile.NamedTemporaryFile(suffix=".cif", delete=False, prefix=f"bm_model{model_num}_")
    tmp.close()
    tmp_path = Path(tmp.name)
    doc.write_file(str(tmp_path))
    return tmp_path


def merge_cif_models(model_paths: list[tuple[int, Path]], output_path: Path) -> None:
    """Merge single-model CIFs into one multi-model CIF.

    Each input CIF contributes one model; ``pdbx_PDB_model_num`` in
    ``_atom_site`` is updated to the supplied model number.  All other CIF
    records (entity, chem_comp, struct_conn, …) are taken from the first model
    — they are topology-level metadata and identical across models that share
    the same sequence.

    Args:
        model_paths: Sequence of (model_num, path) pairs in desired order.
        output_path: Destination multi-model CIF file.

    Raises:
        ValueError: If any input CIF lacks an ``_atom_site`` loop or the
            ``pdbx_PDB_model_num`` column.
    """
    import gemmi

    if not model_paths:
        raise ValueError("No models to merge")

    first_num, first_path = model_paths[0]
    doc = gemmi.cif.read(str(first_path))
    out_block = doc.sole_block()

    loop_ref = out_block.find_loop("_atom_site.id")
    if not loop_ref:
        raise ValueError(f"No _atom_site loop in {first_path}")
    loop = loop_ref.get_loop()
    tags = list(loop.tags)
    n_cols = loop.width()

    model_tag = "_atom_site.pdbx_PDB_model_num"
    if model_tag not in tags:
        raise ValueError(f"{model_tag} column missing from {first_path}")
    model_col = tags.index(model_tag)

    # Build column-major layout for first model, stamp model number.
    # loop.values is flat row-major; set_all_values() expects column-major.
    first_vals = list(loop.values)
    n_rows = len(first_vals) // n_cols
    columns: list[list[str]] = [
        [first_vals[r * n_cols + c] for r in range(n_rows)] for c in range(n_cols)
    ]
    for r in range(n_rows):
        columns[model_col][r] = str(first_num)

    # Append subsequent models column by column
    for model_num, path in model_paths[1:]:
        other_doc = gemmi.cif.read(str(path))
        other_block = other_doc.sole_block()
        lr = other_block.find_loop("_atom_site.id")
        if not lr:
            raise ValueError(f"No _atom_site loop in {path}")
        other_loop = lr.get_loop()
        other_tags = list(other_loop.tags)
        other_n_cols = other_loop.width()
        if model_tag not in other_tags:
            raise ValueError(f"{model_tag} column missing from {path}")

        # Column order is a property of the writer, not of the format, so
        # resolve each of the first model's columns by tag rather than by
        # position: matching on index alone silently interleaves fields (say
        # Cartn_y into Cartn_x) whenever two inputs order _atom_site
        # differently.
        missing = [tag for tag in tags if tag not in other_tags]
        if missing:
            raise ValueError(
                f"{path} is missing _atom_site columns present in {first_path}: "
                f"{', '.join(missing)}"
            )
        other_cols = [other_tags.index(tag) for tag in tags]

        other_vals = list(other_loop.values)
        other_n_rows = len(other_vals) // other_n_cols
        for c, other_c in enumerate(other_cols):
            if c == model_col:
                col_vals = [str(model_num)] * other_n_rows
            else:
                col_vals = [other_vals[r * other_n_cols + other_c] for r in range(other_n_rows)]
            columns[c].extend(col_vals)

    loop.set_all_values(columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.write_file(str(output_path))


def get_residue_info(pdb_path: str | Path) -> list[dict]:
    """Get information about residues in the structure.

    Args:
        pdb_path: Path to PDB file

    Returns:
        List of dictionaries with residue information
    """
    from openmm.app import PDBFile

    pdb = PDBFile(str(pdb_path))
    topology = pdb.topology

    residues = []
    for residue in topology.residues():
        residues.append(
            {
                "name": residue.name,
                "index": residue.index,
                "chain": residue.chain.id,
                "n_atoms": len(list(residue.atoms())),
            }
        )

    return residues
