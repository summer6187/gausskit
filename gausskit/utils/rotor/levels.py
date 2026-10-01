"""Periodic-rotor quantum levels for the rigid-rotor model.

``rotor_levels`` is the gold-standard 1D DVR for a curvilinear fragment rotation (ported
verbatim from mode_aware_toolkit/dos.py:_rotor_levels): a small cosine fit of V(theta) in
a plane-wave basis with the PHYSICAL rotational constant B = KAPPA/I.

``rotor_to_hrd`` is the rigid-rotor analogue of ``dos.scan_to_hrd``: it fits the same
cosine series in the PHYSICAL angle and keeps the physical B (not the scan-length kinetic
B that scan_to_hrd derives for a rectilinear coordinate), returning the dict shape
``dos.hrd_block`` expects so the MultiWell Vhrd2/Bhrd1 deck text renders directly and its
levels reproduce ``rotor_levels``.
"""
import numpy as np

from gausskit.multiwell.dos import _hrd_levels
from gausskit.utils.rotor.constants import KAPPA


def _cosfit(theta_deg, V_cm):
    """Least-squares cosine series a[k] of V over the physical angle; K = min(4, npts//2).

    Cosine-only (no sine): V(theta) is assumed EVEN about theta=0, and the scanned window
    (default +-90 deg) is taken to define one symmetric branch of a 2*pi-periodic rotor. An
    asymmetric torsion, or one whose true period differs from the window, would be mis-modeled.
    """
    phi = np.radians(np.asarray(theta_deg, float))
    V = np.asarray(V_cm, float)
    Kf = min(4, len(phi) // 2)
    a, *_ = np.linalg.lstsq(np.column_stack([np.cos(k * phi) for k in range(Kf)]), V, rcond=None)
    return a


def rotor_levels(theta_deg, V_cm, B, nmax=250):
    """Periodic-rotor DVR levels (cm^-1, ground-referenced) and ground energy E0.

    -B d2/dphi2 + sum_k a[k] cos(k*phi) in the plane-wave basis m = -nmax..nmax. Returns
    ``(levels_ground_referenced, E0)``; E0 is the rotor ground energy (used as its ZPE
    contribution)."""
    a = _cosfit(theta_deg, V_cm)
    ms = np.arange(-nmax, nmax + 1)
    H = np.diag(B * ms.astype(float) ** 2 + a[0])
    for k in range(1, len(a)):
        for i, _ in enumerate(ms):
            if i + k < len(ms):
                H[i, i + k] += a[k] / 2.0
                H[i + k, i] += a[k] / 2.0
    E = np.linalg.eigvalsh(H)
    return E - E[0], float(E[0])


def rotor_to_hrd(theta_deg, V_cm, B, nsym=1):
    """Represent a curvilinear rotation as a MultiWell general hindered rotor (HRD), with the
    PHYSICAL B. Returns ``{"B", "I_red"=KAPPA/B, "CV", "nsym", "levels"}`` (dos.hrd_block-ready).
    ``CV`` are the cosine coefficients a[k]; the HRD levels reproduce ``rotor_levels``."""
    # The physical B = KAPPA/I is the SAME kinetic operator scan_to_hrd builds as
    # C_KIN_EV*(2pi/L)^2*CM_PER_EV: for a rotation Q = sqrt(I)*theta a full turn spans
    # L = 2*pi*sqrt(I), which collapses that expression to (C_KIN_EV*CM_PER_EV)/I == KAPPA/I.
    # Using the TRUE inertia I is why a rectilinear scan cannot over-stiffen the rotation.
    a = _cosfit(theta_deg, V_cm)
    return {"B": float(B), "I_red": float(KAPPA / B), "CV": np.asarray(a, float),
            "nsym": int(nsym), "levels": _hrd_levels(B, a, nsym)}
