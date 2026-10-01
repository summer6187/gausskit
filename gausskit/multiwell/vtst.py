"""gausskit-native microcanonical variational TST (mu-VTST) capture-rate engine.

This is a Python reimplementation of the microcanonical, J-resolved variational
transition-state theory that MultiWell's ``ktools`` performs, built so that the
kinetics workflow can BYPASS the ``ktools`` Fortran and swap the internal harmonic
Beyer-Swinehart state count for an EXTERNAL anharmonic density of states (a bdens
``.dens``).  It is validated by reproducing native ``ktools`` in the harmonic limit
(GATE 3a) on the exact same deck; only the vibrational sum-of-states backend then
changes for the anharmonic result (GATE 3b).

Design (minimise divergence from ktools):
  * The engine consumes the SAME ``ktools.dat`` deck that ``ktools`` reads -- gausskit
    already generates it (energies, ZPE, symmetry, OH spin-orbit ladder, RC-mode
    dropping, mominert moments).  So the ONLY difference in a GATE-3a comparison is
    the solver, not the input.  ``read_ktools_deck`` parses that deck.
  * Per dividing surface the rovibrational sum of states N(E,J) is built from the
    vibrational count (harmonic Stein-Rabinovitch, or a bdens ``.dens`` Sum column),
    the 1D active K-rotor (``kro`` = I_z) and the 2D adiabatic rotor (``jro`` = I_2D)
    whose centrifugal term shifts the effective potential.
  * The variational bottleneck is the surface minimising N(E,J); k_cap(T) is the
    E,J thermal integral over that minimum, normalised by the reactant (fragment)
    partition function.

The physics (state counting, centrifugal V_eff, variational min, E,J integration)
is filled in against a precise reading of the ktools Fortran; see ``_spec`` notes in
the counting routines.  This top section (deck + .dens parsing) is convention-free.
"""

import re
from pathlib import Path

import numpy as np

# --- constants (match dos.py / MultiWell) ------------------------------------
KB_CM = 0.6950348004          # Boltzmann constant, cm^-1 / K
B_FROM_I = 16.85763           # rotational constant B(cm^-1) = B_FROM_I / I(amu*A^2)
CLIGHT = 2.99792458e10        # cm / s
H_ERG_S = 6.62607015e-27      # Planck constant, erg*s  (cgs)
KB_ERG = 1.380649e-16         # Boltzmann constant, erg / K
AMU_G = 1.66053906660e-24     # amu -> g


# =============================================================================
# Deck + DOS parsing  (convention-independent)
# =============================================================================
def _num(tok):
    """Parse a Fortran-ish real/int token, tolerating a trailing '*'."""
    return float(tok.rstrip("*"))


def read_ktools_deck(path):
    """Parse a ``ktools.dat`` deck into a structured dict.

    Returns a dict with the global header and a list of per-structure blocks::

        {
          "units": "KCAL", ...,
          "emax": 40000.0, "egrain": 2.5, "jmax": 500, "jgrain": 1,
          "temps": np.array([...]),
          "nreac": 1, "ntts": 5, "nprod": 2,
          "structures": [ {block}, ... ],
        }

    Each block::

        {
          "type": "reac"|"ctst"|"prod", "name": str,
          "delh": float,          # 0 K energy on the deck's reference (kcal/mol here)
          "dist": float,          # reaction-coordinate distance (A)
          "sigma": int, "optical": int, "nele": int,
          "elev": [(E_cm, g), ...],   # electronic ladder
          "vib":  [w_cm, ...],        # harmonic frequencies (RC mode already dropped)
          "xe":   [xe, ...],          # per-vib Morse anharmonicity (0.0 for HAR)
          "kro":  I_z or None,        # 1D active K-rotor moment (amu*A^2)
          "jro":  I_2D or None,       # 2D adiabatic rotor moment (amu*A^2)
        }

    The block grammar mirrors ktools ``read_input.f``: ``type name delh dist`` /
    formula / 3 comment lines / ``sigma optical nele`` / nele electronic levels /
    ``Ndof HAR AMUA`` / Ndof DOF lines (``vib``/``kro``/``jro``).
    """
    lines = [ln.rstrip("\n") for ln in Path(path).read_text().splitlines()]
    # ---- global header (9 lines) --------------------------------------------
    hdr = [ln for ln in lines]
    title = hdr[0]
    units = hdr[1].split()[0]
    whatdo = hdr[2].strip()
    emax, egrain = (float(x) for x in hdr[3].split()[:2])
    jmax, jgrain = (int(float(x)) for x in hdr[4].split()[:2])
    imax1, isize, emax2 = hdr[5].split()[:3]
    nt = int(hdr[6].split()[0])
    temps = np.array([float(x) for x in hdr[7].split()][:nt])
    nreac, ntts, nprod = (int(float(x)) for x in hdr[8].split()[:3])

    structures = []
    i = 9
    n = len(lines)
    kinds = {"reac", "ctst", "prod"}
    while i < n:
        toks = lines[i].split()
        if not toks or toks[0] not in kinds:
            i += 1
            continue
        blk = {"type": toks[0], "name": toks[1],
               "delh": _num(toks[2]), "dist": _num(toks[3])}
        i += 1
        blk["formula"] = lines[i].strip()        # formula line (e.g. 'HO', 'CO')
        i += 1
        i += 3                                   # 3 comment lines
        sig = lines[i].split()
        blk["sigma"], blk["optical"], blk["nele"] = (int(float(x)) for x in sig[:3])
        i += 1
        elev = []
        for _ in range(blk["nele"]):
            e, g = lines[i].split()[:2]
            elev.append((float(e), float(g)))
            i += 1
        blk["elev"] = elev
        dof = lines[i].split()                    # "Ndof HAR AMUA"
        ndof = int(dof[0])
        i += 1
        vib, xe, kro, jro = [], [], None, None
        for _ in range(ndof):
            d = lines[i].split()
            kind = d[1]
            val = _num(d[2])
            if kind == "vib":
                vib.append(val)
                xe.append(_num(d[3]) if len(d) > 3 else 0.0)
            elif kind == "kro":
                kro = val
            elif kind == "jro":
                jro = val
            i += 1
        blk["vib"], blk["xe"], blk["kro"], blk["jro"] = vib, xe, kro, jro
        structures.append(blk)

    return {"title": title, "units": units, "whatdo": whatdo,
            "emax": emax, "egrain": egrain, "jmax": jmax, "jgrain": jgrain,
            "imax1": imax1, "isize": isize, "emax2": emax2,
            "temps": temps, "nreac": nreac, "ntts": ntts, "nprod": nprod,
            "structures": structures}


def read_dens_sum(path):
    """Read a bdens/densum ``.dens`` file's DOS table.

    Returns ``(E, density, sos)`` numpy arrays (cm^-1, states/cm^-1, cumulative
    sum of states).  Matches both the DENSUM/bdens ``No.  (cm-1)  Density  Sum``
    header and the gausskit ``write_dens`` ``No.  E-Emin  Density  Sum`` header
    (identical 4-column layout) so the harmonic bdens DOS and the scan-rotor
    ``stein_rabinovitch``+``write_dens`` DOS are interchangeable.  Energies are at
    the TOP of each grain, referenced to the ground state.
    """
    lines = Path(path).read_text().splitlines()
    start = next(k for k, ln in enumerate(lines)
                 if re.search(r"No\.\s+(?:\(cm-1\)|E-Emin)\s+Density\s+Sum", ln))
    E, dens, sos = [], [], []
    for ln in lines[start + 1:]:
        t = ln.split()
        if len(t) < 4:
            break
        try:
            idx, e, d, s = int(t[0]), float(t[1]), float(t[2]), float(t[3])
        except ValueError:
            break
        # DENSUM/bdens prints TWO tables -- a fine grid (egrain1) then a redundant coarse grid
        # that RESETS to E=0. Stop at the first block so E stays monotonic (else interp/integrals
        # break: the coarse table's E=0 reset gives a negative dE and a garbage partition function).
        if E and e <= E[-1]:
            break
        E.append(e); dens.append(d); sos.append(s)
    return np.array(E), np.array(dens), np.array(sos)


def kcal_to_cm(x):
    """kcal/mol -> cm^-1."""
    return x * 349.7550881


def b_rot(I_amuA2):
    """Rotational constant B (cm^-1) from a moment of inertia (amu*A^2)."""
    return B_FROM_I / I_amuA2


# =============================================================================
# Microcanonical J-resolved mu-VTST core
# =============================================================================
from gausskit.multiwell.dos import stein_rabinovitch  # noqa: E402

_ELEMENT_MASS = {"H": 1.007825, "C": 12.0, "N": 14.003074, "O": 15.994915,
                 "F": 18.998403, "S": 31.972071, "Cl": 34.968853}


def formula_mass(formula):
    """Total mass (amu) of a simple element-count formula string, e.g. 'HO', 'CO2'."""
    m = 0.0
    for el, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if el:
            m += _ELEMENT_MASS[el] * (int(cnt) if cnt else 1)
    return m


def q_electronic(elev, T):
    """Electronic partition function Sum g_i exp(-E_i/kT) from a [(E_cm, g), ...] ladder."""
    return sum(g * np.exp(-e / (KB_CM * T)) for e, g in elev)


def vib_sos(block, grain, emax, dens_path=None):
    """Vibrational sum of states N_vib(E) for a dividing surface, ground-referenced.

    Two interchangeable backends (identical grid/convention -> harmonic-limit validation
    transfers to the anharmonic result):
      * harmonic  (dens_path is None): exact Beyer-Swinehart/Stein-Rabinovitch convolution
        of the block's harmonic frequencies.
      * anharmonic (dens_path given): the Sum column of a bdens ``.dens``, resampled onto
        the uniform (grain, emax) grid.

    Returns ``(E_grid, sos)`` on ``np.arange(0, emax+grain, grain)``.
    """
    Eg = np.arange(0.0, emax + grain, grain)
    if dens_path is not None:
        Ed, _dens, sos_d = read_dens_sum(dens_path)
        sos = np.interp(Eg, Ed, sos_d, left=0.0, right=sos_d[-1])
        return Eg, sos
    # harmonic ladders, each referenced to its own ground state
    level_sets = [np.arange(0.0, emax + w, w) for w in block["vib"]]
    E_top, _dens, sos_fine = stein_rabinovitch(level_sets, grain, emax + grain)
    sos = np.interp(Eg, E_top, sos_fine)
    return Eg, sos


def surface_NEJ(Eg, sos_vib, V_cm, B_K, B_2D, Jarr,
                krotor_energy="symtop"):
    """Rovibrational sum of states N(E,J) for one dividing surface (electronic factor OUT).

    E is referenced to the PRODUCT (fragment) asymptote; V_cm is the surface's electronic+ZPE
    energy on that reference (cm^-1).  Per total J the 2D adiabatic rotor adds the centrifugal
    barrier to the effective potential, and the 1D active K-rotor is summed as an internal mode::

        V_eff(J)   = V_cm + B_2D * J(J+1)
        E_K(K)     = (B_K - B_2D) * K^2      [symmetric-top, K = 0..J, deg 2 for K>0]  ('symtop')
                   or  B_K * K^2             [free 1D K-rotor]                          ('free')
        N(E,J)     = Sum_K deg(K) * N_vib(E - V_eff(J) - E_K(K))

    Returns an array ``N[j, e]`` over ``Jarr`` x ``Eg``.  ``krotor_energy`` selects the K-rotor
    convention (a tunable pending the ktools cross-check; default symmetric-top).
    """
    grain = Eg[1] - Eg[0]
    nE = len(Eg)
    N = np.zeros((len(Jarr), nE))
    emax = Eg[-1]
    for ij, J in enumerate(Jarr):
        Veff = V_cm + B_2D * J * (J + 1.0)
        if Veff > emax:
            continue                     # centrifugal barrier above the grid -> no open states
        Ks = np.arange(0, J + 1)
        if krotor_energy == "free":
            ek = B_K * Ks * Ks
        else:                            # symmetric top
            ek = (B_K - B_2D) * Ks * Ks
        deg = np.where(Ks == 0, 1.0, 2.0)
        acc = np.zeros(nE)
        for e_k, g_k in zip(ek, deg):
            shift = Veff + e_k
            if shift > emax:
                break                    # ek increases with K -> rest are above grid
            src = np.rint((Eg - shift) / grain).astype(int)
            ok = src >= 0
            acc[ok] += g_k * sos_vib[np.clip(src[ok], 0, nE - 1)]
        # 2-D adiabatic rotor: (2J+1) degeneracy multiplier (ktools sterabj:1713 --
        # NOT convolved; the K-rotor sum above already gives the K-degeneracy).
        N[ij] = (2.0 * J + 1.0) * acc
    return N


def capture_rate(deck, dens_map=None, grain=5.0, emax=None, jmax=None,
                 krotor_energy="symtop", zpe_shift=None, frag_dens=None):
    """Microcanonical J-resolved variational capture rate k_cap(T) from a parsed ktools deck.

    Builds N(E,J) for every trial dividing surface, takes the variational minimum
    N_min(E,J) = min_surface N(E,J) per (E,J), and integrates::

        k_cap(T) = c * Q_elec_TS(T) / Q_reac(T) * Sum_J Sum_E N_min(E,J) e^{-E/kT} * grain

    with Q_reac the product-fragment partition function per unit volume (relative translation
    x rotation x vibration x electronic).  Energies referenced to the product asymptote.

    Args:
        deck: dict from ``read_ktools_deck``.
        dens_map: optional {surface_name: dens_path} to use the anharmonic backend on those
            surfaces (harmonic elsewhere).  None -> fully harmonic (GATE-3a validation mode).
        grain, emax, jmax: integration grid (defaults: deck egrain*?, deck emax, deck jmax).
        krotor_energy: 'symtop' (default) or 'free'.
        zpe_shift: optional {species_name: dZPE_cm} added to that species' delh so the PES is
            referenced to the ANHARMONIC ZPE (dZPE = ZPE_anh - ZPE_harm; see zpe_shift_from_chi).
            Required for a consistent anharmonic run -- the .dens is anharmonic-ground-referenced,
            so its placement must use the anharmonic ZPE (else the landmine-#10 mismatch).
        frag_dens: optional {fragment_name: dens_path} -> the fragment's VPT2 anharmonic q_vib
            replaces the harmonic product, so numerator and Q_reac are both anharmonic.

    Returns dict: T, k_cap (cm^3/molecule/s), r_var (A, canonical bottleneck per T),
    and the per-surface canonical flux for diagnostics.
    """
    emax = float(deck["emax"]) if emax is None else emax
    jmax = int(deck["jmax"]) if jmax is None else jmax
    Eg = np.arange(0.0, emax + grain, grain)
    Jarr = np.arange(0, jmax + 1)
    temps = deck["temps"]

    reac = [s for s in deck["structures"] if s["type"] == "reac"]
    ctst = [s for s in deck["structures"] if s["type"] == "ctst"]
    prod = [s for s in deck["structures"] if s["type"] == "prod"]

    # anharmonic-ZPE-referenced PES: delh_anh = delh_harm + dZPE (cm^-1) per species.
    # zpe_shift={} -> harmonic reference (backward compatible).
    zpe_shift = zpe_shift or {}
    def delh_cm(s):
        return kcal_to_cm(s["delh"]) + zpe_shift.get(s["name"], 0.0)
    Vprod = sum(delh_cm(p) for p in prod)          # fragment asymptote, in cm^-1

    # --- per-surface N(E,J) (electronic factor kept separate) ----------------
    surf_N, surf_dist = [], []
    for s in ctst:
        V_cm = delh_cm(s) - Vprod
        B_K = b_rot(s["kro"])
        B_2D = b_rot(s["jro"])
        dens_path = None if dens_map is None else dens_map.get(s["name"])
        _, sos_vib = vib_sos(s, grain, emax, dens_path=dens_path)
        surf_N.append(surface_NEJ(Eg, sos_vib, V_cm, B_K, B_2D, Jarr,
                                   krotor_energy=krotor_energy))
        surf_dist.append(s["dist"])
    surf_N = np.array(surf_N)                       # [nsurf, nJ, nE]
    N_min = surf_N.min(axis=0)                      # microcanonical variational min

    # TS electronic ladder (all ctst share it here); reactant Q_elec from the fragments
    ts_elev = ctst[0]["elev"]
    m_prod = [formula_mass(p["formula"]) for p in prod]
    mu = m_prod[0] * m_prod[1] / (m_prod[0] + m_prod[1]) if len(m_prod) == 2 else None

    out = {"T": [], "k_cap": [], "r_var": []}
    for T in temps:
        beta = 1.0 / (KB_CM * T)
        boltz = np.exp(-Eg * beta)                  # [nE]
        # numerator: Sum_J Sum_E N_min e^{-E/kT} grain
        num = float((N_min * boltz[None, :]).sum() * grain)
        qelec_ts = q_electronic(ts_elev, T)
        Qreac = _q_reac(prod, deck, mu, T, frag_dens=frag_dens)
        k = CLIGHT * qelec_ts * num / Qreac
        # canonical bottleneck: which surface minimises its own integrated flux at this T
        flux = [float((surf_N[i] * boltz[None, :]).sum()) for i in range(len(ctst))]
        r = surf_dist[int(np.argmin(flux))]
        out["T"].append(float(T)); out["k_cap"].append(k); out["r_var"].append(r)
    for k in out:
        out[k] = np.array(out[k])
    return out


def _prod_formula(deck, block):
    """The formula line stored for a product block (the line after ``type name delh dist``)."""
    return block.get("formula", block["name"])


def _q_reac(prod, deck, mu, T, frag_dens=None):
    """Product-fragment reactant partition function per unit volume (cm^-3), for the reverse
    (capture) direction: relative translation x (rotation x vibration x electronic) of each
    fragment.  A fragment carrying only ``jro`` is treated as a linear 2D rotor; one carrying
    both ``kro`` and ``jro`` as a symmetric top (matches ktools qroq/tqstop).  If ``frag_dens``
    maps a fragment name to a bdens .dens, that fragment's VPT2 anharmonic q_vib is used instead
    of the harmonic product (so numerator and Q_reac are both anharmonic)."""
    frag_dens = frag_dens or {}
    # relative translation (per volume), cm^-3
    qtrans = (2 * np.pi * mu * AMU_G * KB_ERG * T / H_ERG_S ** 2) ** 1.5
    Q = qtrans
    for p in prod:
        # rotation (exact quantum sums; OH has B~19 cm-1 where classical kT/B fails at low T)
        if p["kro"] is not None and p["jro"] is not None:
            Q *= q_symtop(b_rot(p["kro"]), b_rot(p["jro"]), T, sigma=p["sigma"])
        elif p["jro"] is not None:
            Q *= q_rot_2d_quantum(b_rot(p["jro"]), T, sigma=p["sigma"])
        # vibration: anharmonic if provided (a .dens path for a polyatomic, or a (w, X) pair for
        # a diatomic where a 1-mode bdens is degenerate), else the harmonic-oscillator product.
        src = frag_dens.get(p["name"])
        if isinstance(src, str):
            Q *= q_vib_anharmonic(src, T)
        elif src is not None:
            w, X = src
            Q *= q_vib_vpt2(float(w), float(X), T)
        else:
            for w in p["vib"]:
                Q *= 1.0 / (1.0 - np.exp(-w / (KB_CM * T)))
        # electronic
        Q *= q_electronic(p["elev"], T)
    return Q


def zpe_shift_from_chi(chi_path):
    """Anharmonic ZPE shift dZPE = ZPE_anh - ZPE_harm = (1/4) Sum_{i<=j} X_ij (cm^-1) over the
    kept modes, from a build_bdens ``.chi`` file (a full symmetric X matrix).  Placing an
    anharmonic ``.dens`` at ``delh + dZPE`` references the PES to the anharmonic ZPE, consistent
    with the .dens's anharmonic-ground-state origin (CLAUDE.md landmine #10)."""
    X = np.atleast_2d(np.loadtxt(chi_path))
    return 0.25 * float(np.triu(X).sum())          # upper triangle incl. diagonal = Sum_{i<=j}


def q_vib_anharmonic(dens_path, T):
    """VPT2 anharmonic vibrational partition function from a bdens ``.dens`` (ground-referenced):
    ``Sum_E rho_anh(E) e^{-E/kT} dE``.  Drop-in replacement for the harmonic product in Q_reac."""
    E, density, _sos = read_dens_sum(dens_path)
    dE = np.gradient(E)                              # handles the DENSUM double grid
    return float(np.sum(density * np.exp(-E / (KB_CM * T)) * dE))


def q_vib_vpt2(w, X, T, vmax=80):
    """Anharmonic q_vib of a SINGLE mode from its VPT2 constants (harmonic freq ``w``, diagonal
    anharmonicity ``X`` = X_ii, both cm^-1): levels E_v = w(v+1/2) + X(v+1/2)^2, ground-referenced,
    summed.  For a diatomic fragment where a 1-mode bdens ``.dens`` is degenerate.  With X<0 (the
    usual Morse case) levels above the dE/dv=0 turnover are unphysical and dropped."""
    v = np.arange(0, vmax + 1, dtype=float)
    if X < 0:
        v_turn = -0.5 * w / X - 0.5                 # dE/dv = w + 2X(v+1/2) = 0
        v = v[v <= max(0.0, v_turn)]
    E = w * (v + 0.5) + X * (v + 0.5) ** 2
    E = E - E[0]
    return float(np.sum(np.exp(-E / (KB_CM * T))))


def q_rot_2d_quantum(B, T, sigma=1.0, jmax=4000):
    """Exact quantum 2D (linear-molecule) rotational partition function,
    ``(1/sigma) Sum_j (2j+1) exp(-B j(j+1)/kT)`` -- matches ktools ``qroq.f`` (idim=2).
    B in cm^-1.  jmax is a convergence ceiling (ample for B >= ~1 cm^-1)."""
    j = np.arange(0, jmax + 1, dtype=float)
    return float(np.sum((2 * j + 1) * np.exp(-B * j * (j + 1) / (KB_CM * T))) / sigma)


def q_symtop(A, B, T, sigma=1.0, jmax=4000):
    """Exact quantum symmetric-top rotational partition function -- matches ktools ``tqstop.f``::

        (1/sigma) Sum_J (2J+1) e^{-B J(J+1)/kT} [ 1 + 2 Sum_{K=1}^{J} e^{-(A-B) K^2/kT} ]

    A = K-rotor constant (from I_z), B = 2D constant (from I_2D), both cm^-1; K capped at J."""
    J = np.arange(0, jmax + 1, dtype=float)
    ab = A - B
    Q = 0.0
    kbt = KB_CM * T
    for j in J.astype(int):
        if j == 0:
            ksum = 1.0
        else:
            Ks = np.arange(1, j + 1)
            ksum = 1.0 + 2.0 * np.sum(np.exp(-ab * Ks * Ks / kbt))
        Q += (2 * j + 1) * np.exp(-B * j * (j + 1) / kbt) * ksum
        if j > 3 and (2 * j + 1) * np.exp(-B * j * (j + 1) / kbt) < 1e-12 * Q:
            break
    return float(Q / sigma)
