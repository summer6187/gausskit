"""Constants and defaults for the rotor utility.

Centralises two values that were previously homeless — ``KAPPA`` (only defined in
mode_aware_toolkit) and ``HARTREE_CM`` (hardcoded in dos.py and extract_scan.py) —
and re-exports the ``gausskit.multiwell.dos`` kinetic/Boltzmann constants so every
rotor module reads them from one place.
"""
from gausskit.multiwell.dos import CM_PER_EV, C_KIN_EV, KB_CM  # noqa: F401  (re-exported)

#: rotational constant prefactor  B[cm^-1] = KAPPA / I[amu*A^2];  KAPPA = h/(8*pi^2*c).
#: Identical to the 16.857629 that scan_to_hrd hardcodes for I_red (guarded in tests).
KAPPA = 16.857629

#: Hartree -> cm^-1 (for reading rotation-scan electronic energies back to a relative PES).
HARTREE_CM = 219474.6314

# --- classification defaults ---
SOFT_CUT = 200.0        # cm^-1: at/above this a mode is stiff -> harmonic (no scan)
ROT_MIN = 0.60          # min rigid-body rotation fraction to auto-route a mode to rigid-rotor
NMAX = 6                # examine this many lowest REAL modes

# --- scan-grid defaults ---
#: rigid-rotor angular grid (deg): symmetric ±90, step 10 -> 19 points.
DEFAULT_THETA_MIN = -90
DEFAULT_THETA_MAX = 90
DEFAULT_DTHETA = 10
#: mode-scan rectilinear grid (A*amu^1/2): dense ±3 @ 0.25 + integer wings to ±11 (make_qgrid).
DEFAULT_QMAX = 11.0
DEFAULT_DENSE = 3.0
DEFAULT_STEP = 0.25

# --- fit defaults ---
DVR_EMAX = 14000.0                 # cm^-1 ceiling for the gold-standard DVR
QCHECK_TEMPS = (300.0, 700.0, 1000.0)
