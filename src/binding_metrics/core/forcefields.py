"""Force field configurations for MD simulations."""

from dataclasses import dataclass
from typing import Literal

from openmm.app import ForceField


@dataclass(frozen=True)
class ForceFieldConfig:
    """Configuration for a force field.

    Attributes:
        name: Identifier for the force field ('amber' or 'charmm')
        protein_ff: Force field file for proteins
        water_model: Water model file
        description: Human-readable description
    """

    name: str
    protein_ff: str
    water_model: str
    description: str


AMBER_CONFIG = ForceFieldConfig(
    name="amber",
    protein_ff="amber14-all.xml",
    water_model="amber14/tip3pfb.xml",
    description="AMBER ff14SB with TIP3P-FB water",
)

CHARMM_CONFIG = ForceFieldConfig(
    name="charmm",
    protein_ff="charmm36.xml",
    water_model="charmm36/water.xml",
    description="CHARMM36m with CHARMM TIP3P water",
)

FORCEFIELD_CONFIGS: dict[str, ForceFieldConfig] = {
    "amber": AMBER_CONFIG,
    "charmm": CHARMM_CONFIG,
}


def get_forcefield(name: Literal["amber", "charmm"] = "amber") -> ForceField:
    """Get an OpenMM ForceField object for the specified force field.

    Args:
        name: Force field to use ('amber' or 'charmm')

    Returns:
        Configured OpenMM ForceField object

    Raises:
        ValueError: If force field name is not recognized
    """
    if name not in FORCEFIELD_CONFIGS:
        valid = ", ".join(FORCEFIELD_CONFIGS.keys())
        raise ValueError(f"Unknown force field '{name}'. Valid options: {valid}")

    config = FORCEFIELD_CONFIGS[name]
    return ForceField(config.protein_ff, config.water_model)


def get_forcefield_config(name: Literal["amber", "charmm"] = "amber") -> ForceFieldConfig:
    """Get the configuration for a force field.

    Args:
        name: Force field to use ('amber' or 'charmm')

    Returns:
        ForceFieldConfig for the specified force field

    Raises:
        ValueError: If force field name is not recognized
    """
    if name not in FORCEFIELD_CONFIGS:
        valid = ", ".join(FORCEFIELD_CONFIGS.keys())
        raise ValueError(f"Unknown force field '{name}'. Valid options: {valid}")

    return FORCEFIELD_CONFIGS[name]
