"""Structure preparation CLI: fix missing atoms, replace non-standard residues, add hydrogens.

Usage:
    binding-metrics-prep --input complex.cif --output cleaned.cif
    binding-metrics-prep --input complex.pdb --output cleaned.pdb --ph 6.0 --keep-water
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix and protonate a structure using PDBFixer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input structure (.pdb, .cif, .mmcif)")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output structure (.pdb, .cif, .mmcif)")
    parser.add_argument("--ph", type=float, default=7.4, help="pH for hydrogen placement")
    parser.add_argument("--keep-water", action="store_true", help="Retain crystallographic water molecules")
    args = parser.parse_args()

    try:
        from binding_metrics.core.system import prep_structure, HAS_PDBFIXER
        from binding_metrics.io.structures import load_structure, save_structure
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if not HAS_PDBFIXER:
        print(
            "error: pdbfixer is required. Install with: pip install binding-metrics[structure]",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.input.exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    topology, positions = load_structure(args.input)
    n_atoms_before = topology.getNumAtoms()
    n_residues_before = topology.getNumResidues()

    topology, positions = prep_structure(topology, positions, ph=args.ph, keep_water=args.keep_water)

    save_structure(topology, positions, args.output, source_path=args.input)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "ph": args.ph,
        "keep_water": args.keep_water,
        "n_atoms_before": n_atoms_before,
        "n_atoms_after": topology.getNumAtoms(),
        "n_residues_before": n_residues_before,
        "n_residues_after": topology.getNumResidues(),
        "n_chains": topology.getNumChains(),
    }
    print(json.dumps(summary, indent=2))
