"""Plot specification registry for binding-metrics-report.

Each entry maps a short name (used in report config JSON) to a PlotSpec that
describes how to render a single metric column as a histogram.

To add a new plot: append a PlotSpec to PLOT_REGISTRY and reference its name
in a report config under "plots".
"""

from dataclasses import dataclass, field


@dataclass
class PlotSpec:
    """Describes how to render one metric column as a histogram.

    Attributes:
        column: DataFrame column name to plot
        title: Human-readable plot title
        xlabel: x-axis label (defaults to title if empty)
        color: Matplotlib color string
        bins: Number of histogram bins
    """
    column: str
    title: str
    xlabel: str = ""
    color: str = "steelblue"
    bins: int = 30

    @property
    def x_label(self) -> str:
        return self.xlabel or self.title


# ---------------------------------------------------------------------------
# Registry — maps config name → PlotSpec
# ---------------------------------------------------------------------------

PLOT_REGISTRY: dict[str, PlotSpec] = {
    # Interface metrics
    "delta_sasa_plot": PlotSpec(
        column="delta_sasa",
        title="ΔSASA",
        xlabel="ΔSASA (Å²)",
        color="steelblue",
    ),
    "delta_g_int_plot": PlotSpec(
        column="delta_g_int",
        title="ΔG_int",
        xlabel="ΔG_int (kcal/mol)",
        color="tomato",
    ),
    "delta_g_int_kj_plot": PlotSpec(
        column="delta_g_int_kJ",
        title="ΔG_int",
        xlabel="ΔG_int (kJ/mol)",
        color="tomato",
    ),
    "polar_area_plot": PlotSpec(
        column="polar_area",
        title="Polar Interface Area",
        xlabel="Area (Å²)",
        color="mediumpurple",
    ),
    "apolar_area_plot": PlotSpec(
        column="apolar_area",
        title="Apolar Interface Area",
        xlabel="Area (Å²)",
        color="sandybrown",
    ),
    "fraction_polar_plot": PlotSpec(
        column="fraction_polar",
        title="Fraction Polar",
        xlabel="Fraction",
        color="mediumseagreen",
    ),
    "hbonds_plot": PlotSpec(
        column="hbonds",
        title="H-bonds",
        xlabel="Count",
        color="cornflowerblue",
        bins=20,
    ),
    "saltbridges_plot": PlotSpec(
        column="saltbridges",
        title="Salt Bridges",
        xlabel="Count",
        color="goldenrod",
        bins=15,
    ),
    "n_interface_residues_peptide_plot": PlotSpec(
        column="n_interface_residues_peptide",
        title="Interface Residues (peptide)",
        xlabel="Count",
        color="lightcoral",
        bins=20,
    ),
    "n_interface_residues_receptor_plot": PlotSpec(
        column="n_interface_residues_receptor",
        title="Interface Residues (receptor)",
        xlabel="Count",
        color="cadetblue",
        bins=20,
    ),
    # RMSD metrics
    "ligand_rmsd_plot": PlotSpec(
        column="ligand_rmsd",
        title="Ligand RMSD",
        xlabel="RMSD (Å)",
        color="darkorange",
    ),
    "receptor_rmsd_plot": PlotSpec(
        column="receptor_rmsd",
        title="Receptor RMSD",
        xlabel="RMSD (Å)",
        color="slategray",
    ),
    # Energy metrics
    "interaction_energy_plot": PlotSpec(
        column="openmm_interaction_energy",
        title="Interaction Energy (OpenMM)",
        xlabel="Energy (kJ/mol)",
        color="indianred",
    ),
    # Geometry
    "radius_of_gyration_plot": PlotSpec(
        column="radius_of_gyration",
        title="Radius of Gyration",
        xlabel="Rg (Å)",
        color="teal",
    ),
    "end_to_end_distance_plot": PlotSpec(
        column="end_to_end_distance",
        title="End-to-End Distance",
        xlabel="Distance (Å)",
        color="orchid",
    ),
}
