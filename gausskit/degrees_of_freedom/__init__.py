from gausskit.degrees_of_freedom.degrees_of_freedom import DOF
from gausskit.degrees_of_freedom.harmonic import Harmonic

__all__ = ["DOF", "Harmonic", "VPT2"]


def __getattr__(name):
    # VPT2 pulls in gausskit.molecules, which imports Harmonic during its own
    # initialization. Import VPT2 lazily so `import gausskit.molecules` does not
    # trigger a circular import through this package's __init__.
    if name == "VPT2":
        from gausskit.degrees_of_freedom.vpt2 import VPT2
        return VPT2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
