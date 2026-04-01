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
    parser.add_argument(
        "--canonicalize", action="store_true",
        help=(
            "Replace non-standard residues with their nearest standard equivalents "
            "(e.g. MSE→MET, SEP→SER). By default they are preserved so they can be "
            "parameterised downstream with GAFF2 (--small-molecules auto in relax)."
        ),
    )
    parser.add_argument(
        "--no-rebuild-zero-coord-atoms", action="store_true",
        help=(
            "Disable detection and rebuild of zero-coordinate placeholder atoms. "
            "By default, atoms at the origin are removed and rebuilt by PDBFixer "
            "(needed for pipelines that output placeholders instead of modelled atoms)."
        ),
    )
    from binding_metrics.cli import add_log_file_arg
    add_log_file_arg(parser)
    args = parser.parse_args()

    from binding_metrics.cli import log_to_file
    with log_to_file(args.log_file):
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

        topology, positions = prep_structure(
            topology, positions, ph=args.ph,
            keep_water=args.keep_water, canonicalize=args.canonicalize,
            rebuild_zero_coord_atoms=not args.no_rebuild_zero_coord_atoms,
        )

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
