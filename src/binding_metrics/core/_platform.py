"""Choice of the OpenMM platform for a run.

OpenMM registers its CUDA and OpenCL platforms when their plugins load, whether or not a device
is visible, a driver matches the toolkit or the kernels compile. The error shows only when a
context is created on the platform. A platform is therefore called usable here when a small
context (two particles, a Verlet integrator) can be created on it, and the choice of a platform
asks that question instead of asking whether the platform is registered.

The result of the probe is kept for the life of the process, one entry per platform name, and
never invalidated: CUDA initialises once per process, so a device that appears or disappears
afterwards, or a changed ``CUDA_VISIBLE_DEVICES``, is not noticed. The lookup of the platform is
not kept, so a name that stops resolving is seen at once.

A probe that reaches the CUDA platform initialises CUDA in the calling process, and a child
forked from that process cannot create a CUDA context (``CUDA_ERROR_NOT_INITIALIZED``). A process
that forks GPU workers therefore must not probe before it starts them.
"""

import logging

logger = logging.getLogger(__name__)

#: Platforms tried by an automatic choice, in order: the first usable one is taken.
AUTO_ORDER = ("CUDA", "OpenCL", "CPU", "Reference")

#: Platform name -> None when a context was created on it, else the text of the OpenMM error.
#: Filled by the first probe of a name and never emptied (see the module docstring).
_probe_results: dict[str, str | None] = {}


def _registered_platform(name: str):
    """The platform OpenMM registered under ``name``; ``OpenMMException`` when there is none."""
    import openmm

    return openmm.Platform.getPlatformByName(name)


def _create_probe_context(platform) -> None:
    """Create, and release, the smallest context OpenMM accepts on ``platform``."""
    import openmm

    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
    del context


def unusable_reason(name: str) -> str | None:
    """Why no context can be created on the platform ``name``, or None when one can.

    The reason is OpenMM's own message: that no such platform is registered, or the error of the
    context creation (for CUDA without a device, ``CUDA_ERROR_NO_DEVICE``). Only OpenMM's errors
    mean "unusable"; any other exception propagates.
    """
    import openmm

    try:
        platform = _registered_platform(name)
    except openmm.OpenMMException as exc:
        return str(exc)
    if name not in _probe_results:
        try:
            _create_probe_context(platform)
        except openmm.OpenMMException as exc:
            _probe_results[name] = str(exc)
            logger.info("OpenMM platform %s is registered but unusable: %s", name, exc)
        else:
            _probe_results[name] = None
    return _probe_results[name]


def usable_platform(name: str) -> bool:
    """True when a context can be created on the OpenMM platform ``name`` (see the module doc)."""
    return unusable_reason(name) is None


def select_platform(preferred: str = "auto"):
    """The OpenMM platform to run on.

    ``"auto"`` takes the first usable platform of ``AUTO_ORDER`` (CUDA, OpenCL, CPU, Reference),
    so a machine whose CUDA plugin has no device runs on the CPU platform. A named platform is
    returned when it is usable and is never replaced by another: when it is not registered, the
    error is OpenMM's, and when it is registered but no context can be created on it, the error
    is an ``openmm.OpenMMException`` that names the platform and carries OpenMM's message.

    Raises:
        openmm.OpenMMException: ``preferred`` is not usable, or no platform at all is.
    """
    import openmm

    if preferred == "auto":
        reasons = {}
        for name in AUTO_ORDER:
            reason = unusable_reason(name)
            if reason is None:
                return _registered_platform(name)
            reasons[name] = reason
        listing = "; ".join(f"{name}: {reason}" for name, reason in reasons.items())
        raise openmm.OpenMMException(f"no OpenMM platform is usable ({listing})")
    platform = _registered_platform(preferred)
    reason = unusable_reason(preferred)
    if reason is not None:
        raise openmm.OpenMMException(f"OpenMM platform {preferred} is not usable: {reason}")
    return platform
