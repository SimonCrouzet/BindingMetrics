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

#: Default interval in ps between saved MD frames in the relaxation protocol.
DEFAULT_MD_SAVE_INTERVAL_PS = 10.0
