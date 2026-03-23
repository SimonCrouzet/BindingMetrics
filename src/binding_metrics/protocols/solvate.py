"""Solvation CLI: add explicit water box and ions for MD simulation.

Usage:
    binding-metrics-solvate --input cleaned.cif --output solvated.pdb
    binding-metrics-solvate --input cleaned.pdb --output solvated.pdb --padding 1.2 --ionic-strength 0.1
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add explicit solvent and ions to a structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input structure (.pdb, .cif, .mmcif)")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output structure (.pdb, .cif, .mmcif)")
    parser.add_argument("--forcefield", choices=["amber", "charmm"], default="amber", help="Force field for solvent parameters")
    parser.add_argument("--padding", type=float, default=1.0, help="Distance in nm between solute and box edge")
    parser.add_argument("--ionic-strength", type=float, default=0.15, help="Salt concentration in M")
    parser.add_argument("--positive-ion", default="Na+", help="Positive ion type")
    parser.add_argument("--negative-ion", default="Cl-", help="Negative ion type")
    from binding_metrics.cli import add_log_file_arg
    add_log_file_arg(parser)
    args = parser.parse_args()

    from binding_metrics.cli import log_to_file
    with log_to_file(args.log_file):
        try:
            from binding_metrics.core.system import solvate
            from binding_metrics.core.system import get_system_info
            from binding_metrics.io.structures import load_structure, save_structure
        except ImportError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)

        if not args.input.exists():
            print(f"error: input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        topology, positions = load_structure(args.input)

        modeller = solvate(
            topology, positions,
            forcefield_name=args.forcefield,
            padding=args.padding,
            ionic_strength=args.ionic_strength,
            positive_ion=args.positive_ion,
            negative_ion=args.negative_ion,
        )

        save_structure(modeller.topology, modeller.positions, args.output, source_path=args.input)

        info = get_system_info(modeller)
        summary = {
            "input": str(args.input),
            "output": str(args.output),
            "forcefield": args.forcefield,
            "padding_nm": args.padding,
            "ionic_strength_M": args.ionic_strength,
            **info,
        }
        print(json.dumps(summary, indent=2))
