"""Density-of-states utilities for non-standard modes.

MultiWell's DENSUM/BDENS accept only separable standard degrees of freedom (harmonic
or Morse oscillators, free/hindered 1D rotors). A *stiffening* anharmonic mode -- e.g.
a loose-TS transitional bend whose true (scanned) 1D potential rises faster than
harmonic -- cannot be represented by any of those (Morse dissociates, a hindered rotor
softens). This module lets such a mode be handled exactly:

  1. `sinc_dvr`         : exact 1D levels on an arbitrary scanned potential.
  2. `stein_rabinovitch`: sum/density of states by direct convolution of arbitrary
                          level sets (the generalized Beyer-Swinehart count).
  3. `write_dens`       : a MultiWell `.dens` file the master equation can ingest
                          (the only external-DOS entry point in the suite).
  4. partition-function helpers (`q_levels`, `q_harmonic`, `q_rotor`).

Convention: mass-weighted normal coordinate Q in A*amu^(1/2) with kinetic energy
T = 1/(2*mass) * Qdot^2 (mass = 1 for a Gaussian normal mode). Energies in cm^-1.
"""
from __future__ import annotations
import numpy as np

CM_PER_EV = 8065.5439
C_KIN_EV = 2.0900800e-3          # hbar^2/2 in eV*amu*A^2
KB_CM = 0.6950348004            # Boltzmann constant in cm^-1/K


# ---- 1D quantum levels on an arbitrary potential -----------------------------
def sinc_dvr(Q, V_cm, mass=1.0, npts=600, emax=30000.0):
    """Colbert-Miller sinc-DVR levels (cm^-1, referenced to the potential minimum)
    for a 1D mode sampled as (Q, V_cm).

    Q in A*amu^(1/2); V_cm in cm^-1; mass = effective mass (1 for a mass-weighted
    normal coordinate). The samples are cubic-spline interpolated (linear fallback
    if SciPy is absent) onto a uniform grid of `npts` points spanning the scan.
    Only bound levels below `emax` are returned.
    """
    Q = np.asarray(Q, float)
    V_cm = np.asarray(V_cm, float)
    order = np.argsort(Q)
    Q, V_cm = Q[order], V_cm[order]
    try:
        from scipy.interpolate import CubicSpline
        f = CubicSpline(Q, V_cm)
    except Exception:
        f = lambda x: np.interp(x, Q, V_cm)
    g = np.linspace(Q[0], Q[-1], npts)
    dq = g[1] - g[0]
    Vg = np.asarray(f(g), float)
    Vmin = Vg.min()
    i = np.arange(npts)
    D = i[:, None] - i[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(D == 0, np.pi**2 / 3.0, 2.0 * (-1.0)**D / np.where(D == 0, 1, D)**2)
    K = K * (C_KIN_EV * CM_PER_EV) / (mass * dq**2)        # kinetic operator, cm^-1
    E = np.linalg.eigvalsh(K + np.diag(Vg)) - Vmin
    return np.sort(E[E < emax])


# ---- generalized Beyer-Swinehart / Stein-Rabinovitch state count -------------
def stein_rabinovitch(level_sets, grain, emax):
    """Sum and density of states by exact convolution of arbitrary 1D level sets.

    `level_sets` is a list of 1-D arrays, each a mode's energy levels (cm^-1); each
    set is internally referenced to its own ground state (its minimum), so DVR levels
    that carry a ZPE offset are folded in correctly. Returns (E_top, density, sos)
    on a uniform grid of spacing `grain` up to `emax`:
      - E_top   : energy at the TOP of each grain (MultiWell convention),
      - sos     : cumulative sum of states (number of states with E <= E_top),
      - density : states per cm^-1 (the per-grain count divided by `grain`).
    """
    n = int(round(emax / grain)) + 1
    counts = np.zeros(n)
    counts[0] = 1.0                                        # the global ground state
    for levels in level_sets:
        levels = np.asarray(levels, float)
        idx = np.rint((levels - levels.min()) / grain).astype(int)
        idx = idx[(idx >= 0) & (idx < n)]
        conv = np.zeros(n)
        for j in idx:
            conv[j:] += counts[:n - j]
        counts = conv
    sos = np.cumsum(counts)
    density = counts / grain
    E_top = np.arange(n) * grain
    return E_top, density, sos


# ---- MultiWell .dens writer --------------------------------------------------
def write_dens(path, title, reference, egrain1, imax1, emax2, density, sos,
               viblo=1.0):
    """Write a MultiWell `.dens` file (read by the master equation as
    DensData/<name>.dens). `density` and `sos` are length-Isize arrays on the
    grid implied by (egrain1, imax1, emax2); the master equation enforces that
    egrain1/imax1/emax2/Isize match its `.dat` deck, so pass the same values.
    Energies are at the TOP of each grain. Returns Isize.
    """
    density = np.asarray(density, float)
    sos = np.asarray(sos, float)
    isize = len(density)
    E = _dens_energies(egrain1, imax1, emax2, isize)
    lines = [
        " " + title,
        " " + reference,
        f"{egrain1:11.1f}{imax1:11d}{emax2:11.1f}{isize:11d}{viblo:11.1f}",
        "       No.    E-Emin     Density     Sum   "
        "[Emin =    0.0 cm-1; E at TOP of energy grains]",
    ]
    for k in range(isize):
        lines.append(f"{k+1:10d}{E[k]:11.1f}  {density[k]:.5E}  {sos[k]:.5E}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return isize


def _dens_energies(egrain1, imax1, emax2, isize):
    """The DENSUM double-grid energies (TOP of grain): `imax1` fine grains of
    `egrain1`, then a coarser grain linearly to `emax2` over the remaining points.
    Single-grain if imax1 >= isize."""
    E = np.arange(isize, dtype=float) * egrain1
    if imax1 < isize:
        coarse = (emax2 - (imax1 - 1) * egrain1) / (isize - imax1)
        E[imax1:] = (imax1 - 1) * egrain1 + np.arange(1, isize - imax1 + 1) * coarse
    return E


def dens_on_grid(level_sets, egrain1, imax1, emax2, isize):
    """Convenience: build (density, sos) on a DENSUM grid from level sets, by
    convolving on the fine grain and sampling onto the (possibly double) grid."""
    E_top = _dens_energies(egrain1, imax1, emax2, isize)
    _, _, sos_fine = stein_rabinovitch(level_sets, egrain1, E_top[-1] + egrain1)
    Ef = np.arange(len(sos_fine)) * egrain1
    sos = np.interp(E_top, Ef, sos_fine)
    density = np.gradient(sos, E_top)
    return density, sos


# ---- partition functions -----------------------------------------------------
def q_levels(levels, T):
    """Vibrational partition function from explicit levels (cm^-1), ground-referenced."""
    levels = np.asarray(levels, float)
    return float(np.exp(-(levels - levels.min()) / (KB_CM * T)).sum())


def q_harmonic(w, T):
    """Harmonic-oscillator partition function (cm^-1 frequency), ground-referenced."""
    return 1.0 / (1.0 - np.exp(-w / (KB_CM * T)))


def q_rotor(B, T, dim=1, sigma=1.0):
    """Classical free-rotor partition function. dim=1: sqrt(pi*kT/B)/sigma;
    dim=2: (kT/B)/sigma. B in cm^-1."""
    x = KB_CM * T / B
    if dim == 1:
        return float(np.sqrt(np.pi * x) / sigma)
    if dim == 2:
        return float(x / sigma)
    raise ValueError("dim must be 1 or 2")
