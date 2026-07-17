"""`gausskit utils rotor` — determine per-soft-mode rotor information.

One utility that unifies the previously scattered rotor machinery (the na-cfxcly
`hindered_rotor_kit`, the ch2o+OH `mode_aware_toolkit`, and the standalone
`gausskit.multiwell.dos` primitives) behind a single hand-editable YAML roadmap.

Two rotor models, chosen per mode by the ``method`` field of ``rotor_plan.yaml``:

  * ``mode-scan``  — rigid rectilinear scan along the normal-mode eigenvector,
    x(Q)=x0+(L/sqrt(m))Q; levels from a bounded-well ``sinc_dvr``.  (na-cfxcly
    stiffening tight-TS wells.)
  * ``rigid-rotor`` — curvilinear rotation of a fragment about its centre of mass;
    levels from a periodic-rotor DVR with the physical B = KAPPA/I.  (ch2o+OH
    librations, which a rectilinear scan over-stiffens.)
  * ``harmonic``   — stiff mode, not scanned.

Workflow (verbs): ``detect`` (classify + write the roadmap) -> hand-edit ->
``scan`` (roadmap -> gjf) -> submit externally -> ``fit`` (roadmap + logs ->
levels, Vhrd2/HRD blocks, validation).
"""
from gausskit.utils.rotor.character import classify_modes, format_table, rot_decomp
from gausskit.utils.rotor.plan import build_plan, dump_plan, load_plan
from gausskit.utils.rotor.scan import emit_scans
from gausskit.utils.rotor.fit import fit_rotors
from gausskit.utils.rotor.result import load_fitted_modes
from gausskit.utils.rotor.patch import patch_pes

__all__ = [
    "classify_modes", "format_table", "rot_decomp",
    "build_plan", "dump_plan", "load_plan",
    "emit_scans", "fit_rotors", "load_fitted_modes", "patch_pes",
]
