"""Per-mode character classifier — the single source of truth for the rotor decision.

Parses geometry + harmonic Cartesian normal modes with
``gausskit.gaussian.log_parser.read_normal_modes`` (the SAME reader/order as
``gausskit.utils.mode_scan.generate_mode_scan``, so a mode's index — 1-based over the
frequency-ascending real modes — is identical across ``detect``/``scan``/``fit``).
For each of the ``nmax`` lowest real modes it decomposes each fragment's raw-Cartesian
normal-mode displacement (treated as a velocity field) into translation + rigid rotation
+ internal, using the normal mode only as a local tangent, and assigns an auto ``method``.
Mass-weighting enters ONLY as the least-squares metric of that decomposition (never
pre-weight ``disp``, or it is double-counted):

    stiff (freq >= soft_cut)                       -> harmonic          (no scan)
    soft, 2 fragments, rot_frac >= rot_min         -> rigid-rotor       (B = KAPPA/I, see below)
    soft, otherwise (incl. 1-fragment tight TS)    -> mode-scan         (rectilinear well)

The recorded ``I`` is a SINGLE fragment's moment about its OWN centre of mass, so
``B = KAPPA/I`` is only an auto-seed for the true reduced moment of the two counter-rotating
fragments (exact when the partner is much heavier / the axis passes through the bond). It is
written into the hand-editable roadmap; override with MultiWell ``mominert``'s reduced moment
for a quantitative run (CLAUDE.md landmine: reduced moments come from mominert, not by hand).

``rot_decomp`` runs on every soft mode so ``axis``/``I``/``rot_frac`` are always
available even when a mode is auto-routed to ``mode-scan`` — the hand-edit escape hatch
(flip ``method`` to ``rigid-rotor`` in the roadmap) then has the params it needs.
"""
import numpy as np

from gausskit.gaussian.log_parser import read_normal_modes
from gausskit.utils.rotor.constants import KAPPA, SOFT_CUT, ROT_MIN, NMAX
from gausskit.utils.rotor.fragments import detect_fragments, reaction_bond

# atomic number -> symbol (covers the elements in fragments.RCOV; extend as needed)
_Z = {1: "H", 2: "He", 3: "Li", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na",
      12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 19: "K", 35: "Br", 53: "I"}


def _symbols(numbers):
    return [_Z.get(int(z), "X") for z in numbers]


def rot_decomp(masses, xyz, disp, idx):
    """Rigid-rotation content of a raw-Cartesian normal-mode displacement over the atom
    subset ``idx``. Ported from mode_aware_toolkit/classify.py:69, generalised to take a
    parsed ``masses`` array (amu) instead of a hardcoded element->mass dict.

    Removes the subset COM translation, fits the rigid angular velocity omega = I^-1 L,
    and returns ``(rot_frac, axis[3], I_about_axis [amu*A^2], mass_weighted_weight)``:
      rot_frac  fraction of the mass-weighted displacement explained by rigid rotation,
      axis      unit rotation axis,
      I         moment of inertia of the subset about that axis,
      weight    total mass-weighted displacement magnitude of the subset (for picking the
                dominant fragment among the two).
    """
    m = np.asarray(masses, float)[idx]
    r = np.asarray(xyz, float)[idx]
    d = np.asarray(disp, float)[idx]
    com = (m[:, None] * r).sum(0) / m.sum()
    rc = r - com
    d0 = d - (m[:, None] * d).sum(0) / m.sum()               # remove COM translation
    Itn = sum(mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri)) for mi, ri in zip(m, rc))
    L = sum(mi * np.cross(ri, di) for mi, ri, di in zip(m, rc, d0))
    omega = np.linalg.solve(Itn + 1e-9 * np.eye(3), L)
    d_rot = np.array([np.cross(omega, ri) for ri in rc])
    frac = min((m[:, None] * d_rot ** 2).sum() / max((m[:, None] * d0 ** 2).sum(), 1e-12), 1.0)
    weight = float((m[:, None] * d ** 2).sum())
    if np.linalg.norm(omega) < 1e-9:
        return 0.0, np.array([0.0, 1.0, 0.0]), 0.0, weight
    axis = omega / np.linalg.norm(omega)
    return float(frac), axis, float(axis @ Itn @ axis), weight


def classify_modes(log, *, soft_cut=SOFT_CUT, rot_min=ROT_MIN, nmax=NMAX, tol=1.3):
    """Classify the ``nmax`` lowest real modes of a Gaussian freq ``log``.

    Returns a dict::

        {"n_fragments", "fragments" (0-based atom lists, largest first),
         "frag_labels" ("A","B",...), "frag_formula", "reaction_bond" (d,i,j)|None,
         "modes": [ {mode, gauss_idx, freq, rot_frac, frag (label|None), axis|None,
                     I, B, auto_method}, ... ]}

    ``gauss_idx`` is the 0-based index into the frequency log's mode order — the anchor
    that keeps detect/scan/fit consistent. ``mode`` is the 1-based rank among real modes.
    """
    numbers, pos, mass, freqs, cart = read_normal_modes(log)
    sym = _symbols(numbers)
    frags = detect_fragments(sym, pos, tol)
    labels = [chr(ord("A") + k) for k in range(len(frags))]
    two_fragments = len(frags) == 2

    # the nmax lowest REAL modes: argsort ascending, then drop the imaginary/TS mode
    # (read_normal_modes stores an imaginary frequency as NEGATIVE, so freqs > 0 filters it out).
    real = [g for g in np.argsort(freqs) if freqs[g] > 0][:nmax]
    modes = []
    for rank, g in enumerate(real, start=1):
        fr = float(freqs[g])
        disp = cart[g]
        # best-fitting fragment rotation (max rot_frac * mass-weighted weight)
        best = None
        for fi, idx in enumerate(frags):
            frac, axis, Iax, w = rot_decomp(mass, pos, disp, idx)
            if best is None or frac * w > best[0]:
                best = (frac * w, fi, frac, axis, Iax)
        _, fi, frac, axis, Iax = best
        # Iax is the rotating fragment's moment about its OWN COM -> B here is an auto-seed
        # for the true reduced moment (see the module docstring); the roadmap is hand-editable.
        B = KAPPA / Iax if Iax > 0.05 else float("nan")

        if fr >= soft_cut:
            auto = "harmonic"                       # stiff -> harmonic (no scan)
        elif two_fragments and frac >= rot_min:
            auto = "rigid-rotor"                    # soft rigid-body fragment rotation
        else:
            # 1-fragment tight TS, >2 fragments, or low rot_frac: no clean fragment
            # rotation -> rectilinear bounded well (NOT harmonic).
            auto = "mode-scan"

        # Always record the best-fragment rotation params (frag/axis/I) so a hand-edit
        # to method=rigid-rotor in the roadmap has everything it needs.
        modes.append({
            "mode": rank, "gauss_idx": int(g), "freq": fr, "rot_frac": frac,
            "frag": labels[fi], "axis": [float(x) for x in axis], "I": float(Iax),
            "B": float(B), "auto_method": auto,
        })

    return {
        "n_fragments": len(frags),
        "fragments": [[int(i) for i in f] for f in frags],
        "frag_labels": labels,
        "frag_formula": ["".join(sym[i] for i in f) for f in frags],
        "reaction_bond": reaction_bond(sym, pos, frags),
        "modes": modes,
    }


def format_table(result):
    """Human-readable decision table (used by ``gausskit utils rotor detect``)."""
    lines = []
    nf = result["n_fragments"]
    frag_str = ", ".join(f"{lab}={f}" for lab, f
                         in zip(result["frag_labels"], result["frag_formula"]))
    lines.append(f"# fragments ({nf}): {frag_str}")
    if nf not in (1, 2):
        lines.append(f"! WARNING: {nf} fragments detected — expected 1 (tight TS) or 2 "
                     "(entrance); check geometry/--tol. All soft modes default to mode-scan.")
    rb = result["reaction_bond"]
    if rb is not None:
        lines.append(f"# reaction bond: atoms {rb[1]}-{rb[2]}  d = {rb[0]:.2f} A")
    lines.append(f"{'mode':>4} {'gidx':>4} {'freq':>8} {'rot_frac':>8} {'frag':>4} "
                 f"{'I':>8} {'B':>8}  method")
    for m in result["modes"]:
        Ii = "   -   " if not np.isfinite(m["I"]) or m["I"] <= 0 else f"{m['I']:7.2f}"
        Bi = "   -   " if not np.isfinite(m["B"]) else f"{m['B']:7.3f}"
        lines.append(f"{m['mode']:>4} {m['gauss_idx']:>4} {m['freq']:8.1f} {m['rot_frac']:8.2f} "
                     f"{str(m['frag'] or '-'):>4} {Ii:>8} {Bi:>8}  {m['auto_method']}")
    return "\n".join(lines)
