"""Force field configurations for MD simulations.

The configuration dataclasses and ``get_forcefield_config`` need nothing beyond
the standard library. OpenMM is imported when ``get_forcefield`` builds a
``ForceField``, so this module imports on installs without OpenMM.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
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


def __getattr__(name: str):
    """Resolve ``ForceField``, which this module used to import at load time (PEP 562)."""
    if name == "ForceField":
        from openmm.app import ForceField

        return ForceField
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_forcefield(name: Literal["amber", "charmm"] = "amber") -> "ForceField":
    """Get an OpenMM ForceField object for the specified force field.

    Args:
        name: Force field to use ('amber' or 'charmm')

    Returns:
        Configured OpenMM ForceField object

    Raises:
        ValueError: If force field name is not recognized
        ModuleNotFoundError: If OpenMM cannot be imported (an ImportError subclass);
            the message names the extra to install
    """
    if name not in FORCEFIELD_CONFIGS:
        valid = ", ".join(FORCEFIELD_CONFIGS.keys())
        raise ValueError(f"Unknown force field '{name}'. Valid options: {valid}")

    try:
        from openmm.app import ForceField
    except ImportError as exc:
        raise ModuleNotFoundError(
            "get_forcefield needs OpenMM, which could not be imported. "
            "Install it with `pip install binding-metrics[simulation]`, "
            "or use environment.yml for a GPU build.",
            name="openmm",
        ) from exc

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
