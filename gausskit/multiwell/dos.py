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


# ---- scan -> MultiWell general hindered rotor (HRD) ---------------------------
def scan_to_hrd(Q, V_cm, n_fourier=16, nsym=1):
    """Represent a scanned 1D mode (Q in A*amu^1/2, V_cm in cm^-1 referenced to the
    minimum) as a MultiWell general hindered rotor (HRD) that reproduces the mode's
    quantum levels. The mass-weighted coordinate Q is mapped onto a torsional angle
    phi in [0, 2*pi) over the scan range L; the kinetic term -C_KIN d2/dQ2 becomes
    -B d2/dphi2 with B = C_KIN*(2*pi/L)^2 (so the rotor's reduced moment is a kinetic
    bookkeeping device, not a physical inertia), and V(phi) is a cosine Fourier series
    (the MultiWell 'Vhrd2' model: V = CV[0] + sum_i CV[i] cos(i*nsym*phi)).

    Returns dict: B (cm^-1), I_red (amu*A^2 = 16.857629/B), CV (cosine coefficients),
    nsym, and the bound levels (cm^-1) the HRD yields (for validation).
    """
    Q = np.asarray(Q, float)
    V_cm = np.asarray(V_cm, float)
    L = Q.max() - Q.min()
    phi = 2 * np.pi * (Q - Q.min()) / L
    B = C_KIN_EV * (2 * np.pi / L) ** 2 * CM_PER_EV
    A = np.column_stack([np.cos(n * nsym * phi) for n in range(n_fourier)])
    CV, *_ = np.linalg.lstsq(A, V_cm - V_cm.min(), rcond=None)
    levels = _hrd_levels(B, CV, nsym)
    return {"B": float(B), "I_red": float(16.857629 / B), "CV": CV,
            "nsym": int(nsym), "levels": levels}


def _hrd_levels(B, CV, nsym=1, mmax=120, emax=20000.0):
    """Eigenvalues (cm^-1, ground-referenced) of -B d2/dphi2 + sum CV[n]cos(n*nsym*phi)
    in the plane-wave basis (exact for a cosine potential)."""
    CV = np.asarray(CV, float)
    m = np.arange(-mmax, mmax + 1)
    n = len(m)
    H = np.diag(B * m.astype(float) ** 2)
    for k in range(1, len(CV)):
        for i in range(n):
            j = i + k * nsym
            if 0 <= j < n:
                H[i, j] += CV[k] / 2.0
                H[j, i] += CV[k] / 2.0
    E = np.linalg.eigvalsh(H)
    E = np.sort(E - E.min())
    return E[E < emax]


def hrd_block(mode_no, hrd, comment="scan-matched hindered rotor"):
    """MultiWell DENSUM/THERMO input lines for a general hindered rotor (HRD) from a
    `scan_to_hrd` result `hrd`. Format:
        <mode>  hrd  <NSV>.  1.  <nsym>   ! comment
          Vhrd2  <nsym>  0.  <CV...>
          Bhrd1  1  0.  <B>
    """
    CV = np.asarray(hrd["CV"], float)
    nsv = len(CV)
    vcoef = "  ".join(f"{c:.5f}" for c in CV)
    return (f"{mode_no:<6} hrd   {nsv}.   1.   {hrd['nsym']}   ! {comment}\n"
            f"  Vhrd2   {hrd['nsym']}   0.   {vcoef}\n"
            f"  Bhrd1   1   0.   {hrd['B']:.6f}")


# ---- SCTST / paradensum bound-state gating for anharmonic (VPT2) manifolds ----
# Faithful NumPy ports of the MultiWell-2023.1 routines that decide which anharmonic states are bound
# and may be counted (src/bdens/nvmax.f, src/parsctst/ckderiv.f), plus the Monte-Carlo helpers used to
# build a thermal bath occupation. Frequencies `w` (N,) and the full anharmonicity matrix `X` (N, N)
# are in cm^-1 over the REAL modes only (drop the imaginary reaction coordinate first; e.g. from
# Molecules.frequencies / Molecules.anharm_matrix). The VPT2 vibrational term value is
#     G(v) = sum_k w_k (v_k+1/2) + sum_{k<=l} x_kl (v_k+1/2)(v_l+1/2)      (cm^-1),
# so the level energy of mode i over a frozen bath {n_l} is
#     dE_i(ni) - dE_i(0) = ni * ( w_i + x_ii (ni+1) + sum_{l!=i} x_il (n_l+1/2) ) .
def nvmax_turnover(w, x, njmax=10):
    """Highest BOUND quantum number of a 1-D anharmonic mode (multiwell-2023.1 src/bdens/nvmax.f).

    For x < 0 the VPT2 ladder folds over at the Birge-Sponer turnover ``vd = -w/(2x) - 1/2`` (return
    ``INT(vd)``); for x >= 0 it never folds over (return the ``njmax`` ceiling); return -1 if the mode
    is unbound (w <= 0).
    """
    if w <= 0.0:
        return -1
    if x < 0.0:
        return max(0, int(-w / (2.0 * x) - 0.5))
    return njmax


def ckderiv_mask(U, w, X):
    """Vectorised bound-state derivative test (multiwell-2023.1 src/parsctst/ckderiv.f).

    A configuration is bound iff every partial derivative
        dG/dv_k = w_k + 2 x_kk (v_k+1/2) + sum_{l!=k} x_kl (v_l+1/2)
    is > 0. ``U`` = v + 1/2 with shape (M, N) (M sampled configurations over N real modes); ``w`` (N,)
    and ``X`` (N, N) in cm^-1. Returns a boolean (M,): True where the configuration is bound. (The
    ``U @ X`` term already contains x_kk U_k, so ``np.diag(X)*U + U@X`` gives the full 2 x_kk term.)
    """
    deriv = w + np.diag(X) * U + U @ X
    return np.all(deriv > 0.0, axis=1)


def bath_coupling_sum(i, X, bath_occ, bath_idx):
    """Soft-mode coupling sum S = sum_{l in bath} x_il (n_l + 1/2) for each sampled bath vector.

    ``bath_occ`` (M, Nbath) integer occupations; ``bath_idx`` their indices into the real-mode set.
    Computed once and reused for every ni, since dE_i(ni) = ni*(w_i + x_ii(ni+1)) + ni*S.
    """
    return (bath_occ + 0.5) @ X[i, bath_idx]


def boltzmann_bath(w, X, bath_idx, T, rng, n_samples, njmax=10):
    """Sample each bath mode independently from its 1-D VPT2 anharmonic Boltzmann weight.

    Per mode j the occupation n is drawn from p(n) propto exp(-E_j(n)/kT), E_j(n) = w_j n +
    x_jj (n^2+n), over n = 0..min(nvmax_turnover, njmax) (a fold-over/unbound mode contributes only
    n=0). Returns integer occupations (n_samples, Nbath). ``rng`` is a caller-seeded numpy Generator.
    """
    cols = np.empty((n_samples, bath_idx.size), dtype=np.int32)
    for c, j in enumerate(bath_idx):
        cap = nvmax_turnover(w[j], X[j, j], njmax)
        cap = min(cap, njmax) if cap >= 0 else 0
        levels = np.arange(cap + 1)
        Ej = w[j] * levels + X[j, j] * (levels ** 2 + levels)         # 1-D anharmonic ladder
        p = np.exp(-(Ej - Ej.min()) / (KB_CM * T))
        cols[:, c] = rng.choice(levels, size=n_samples, p=p / p.sum())
    return cols


def vpt2_thermal_levels(w, X, i, T, rng, nimax=12, n_samples=100_000, njmax=10):
    """Surviving coupled VPT2 level energies E(ni)-E(0) for soft mode ``i`` under the paradensum/sctst
    bound-state gating at temperature ``T`` -- the orchestration that ties the gating primitives above
    together for one mode.

    ``i`` indexes the real-mode arrays ``w`` (Nr,) and ``X`` (Nr, Nr). The bath (all real modes != i)
    is Boltzmann-sampled (``boltzmann_bath``, capped at ``nvmax_turnover``); for each ni = 0..nimax the
    coupled level
        dE_i(ni) = ni * ( w_i + x_ii (ni+1) + sum_{l!=i} x_il (n_l+1/2) )
    is kept only where ``ckderiv_mask`` passes AND dE >= 0. Returns a list of length nimax+1; element
    ni is the 1-D array of surviving dE samples -- its mean is the effective (counted) level and its
    size / n_samples the surviving fraction. ``rng`` is a caller-seeded numpy Generator.
    """
    nr = len(w)
    wi, xii = w[i], X[i, i]
    bath_idx = np.array([j for j in range(nr) if j != i])
    bath_occ = boltzmann_bath(w, X, bath_idx, T, rng, n_samples, njmax)
    S = bath_coupling_sum(i, X, bath_occ, bath_idx)
    U = np.full((n_samples, nr), 0.5)
    U[:, bath_idx] = bath_occ + 0.5
    out = []
    for ni in range(nimax + 1):
        U[:, i] = ni + 0.5
        dE = ni * (wi + xii * (ni + 1.0)) + ni * S
        out.append(dE[ckderiv_mask(U, w, X) & (dE >= 0.0)])
    return out
