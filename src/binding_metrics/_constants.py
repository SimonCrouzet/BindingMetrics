"""Package-wide constants that must be importable without OpenMM."""

#: Default seed for every stochastic step (hydrogen placement, PDBFixer atom
#: rebuild, MD velocities and Langevin noise) so the pipeline is reproducible by
#: default. MUST be non-zero: OpenMM's ``setRandomNumberSeed(0)`` is the sentinel
#: for "choose a fresh random seed at run time", so a 0 here would silently
#: re-randomize the integrators it is fed to (e.g. PDBFixer.addMissingAtoms).
#: The specific value carries no meaning; do not "tune" it to dodge a bad
#: hydrogen placement — repair_ca_hydrogen_chirality exists to fix those. Pass
#: ``random_seed=None`` through the configs to opt back into fresh randomness.
DEFAULT_RANDOM_SEED = 1

#: Default pH for hydrogen placement (structure prep, relaxation, interaction energy).
DEFAULT_PH = 7.4

#: Default compute device of the OpenMM steps.
DEFAULT_DEVICE = "cuda"

#: Default duration in ps of the relaxation MD run; 0 minimises only.
DEFAULT_MD_DURATION_PS = 200.0

#: Default interval in ps between saved MD frames in the relaxation protocol.
DEFAULT_MD_SAVE_INTERVAL_PS = 10.0

#: Coulomb constant e^2 N_A / (4 pi eps0) in kJ A / (mol e^2) (CODATA 2018 gives
#: 1389.354576). The one value of the package: the Coulomb metric, the salt-bridge
#: score and the pair sums of the interaction energy all derive from it.
COULOMB_KJ_ANG_MOL_E2 = 1389.35457

_KJ_PER_KCAL = 4.184  # thermochemical calorie, exact by definition

#: The same constant in kcal A / (mol e^2).
COULOMB_KCAL_ANG_MOL_E2 = COULOMB_KJ_ANG_MOL_E2 / _KJ_PER_KCAL

#: The same constant in kJ nm / (mol e^2), the unit of OpenMM.
COULOMB_KJ_NM_MOL_E2 = COULOMB_KJ_ANG_MOL_E2 / 10.0
