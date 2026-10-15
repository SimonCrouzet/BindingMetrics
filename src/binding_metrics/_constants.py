"""Package-wide constants that must be importable without OpenMM."""

#: Default seed for every stochastic step (hydrogen placement, PDBFixer atom
#: rebuild, MD velocities and Langevin noise). Must be non-zero: OpenMM's
#: ``setRandomNumberSeed(0)`` means "choose a fresh seed at run time". Pass
#: ``random_seed=None`` through the configs to turn seeding off.
DEFAULT_RANDOM_SEED = 1

#: Default pH for hydrogen placement (structure prep, relaxation, interaction energy).
DEFAULT_PH = 7.4

#: Default compute device of the OpenMM steps.
DEFAULT_DEVICE = "cuda"

#: Default duration in ps of the relaxation MD run; 0 minimises only.
DEFAULT_MD_DURATION_PS = 200.0

#: Default interval in ps between saved MD frames in the relaxation protocol.
DEFAULT_MD_SAVE_INTERVAL_PS = 10.0

#: Coulomb constant e^2 N_A / (4 pi eps0) in kJ A / (mol e^2) (CODATA 2018: 1389.354576).
#: The Coulomb metric, the salt-bridge score and the interaction-energy pair sums use it.
COULOMB_KJ_ANG_MOL_E2 = 1389.35457

_KJ_PER_KCAL = 4.184  # thermochemical calorie, exact by definition

#: The same constant in kcal A / (mol e^2).
COULOMB_KCAL_ANG_MOL_E2 = COULOMB_KJ_ANG_MOL_E2 / _KJ_PER_KCAL

#: The same constant in kJ nm / (mol e^2), the unit of OpenMM.
COULOMB_KJ_NM_MOL_E2 = COULOMB_KJ_ANG_MOL_E2 / 10.0
