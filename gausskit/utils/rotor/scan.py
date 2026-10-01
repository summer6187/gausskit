"""Emit scan geometry from the roadmap — dispatch per mode, NO re-classification.

For each mode in ``rotor_plan.yaml``:
  * ``rigid-rotor`` -> rotate the named fragment about its centre of mass about the roadmap
    ``axis`` (Rodrigues) over the theta grid -> mode{k}/r_XX.gjf (title '... theta=NN deg').
  * ``mode-scan``  -> rigid rectilinear scan of the eigenvector (``gauss_idx``) over the Q
    grid via ``emit_rectilinear_mode`` -> mode{k}/sm{k}_XX.gjf (title '... Q=+X.XXX').
  * ``harmonic``   -> skipped.
The title conventions are exactly what ``rotor.readback`` parses back at the fit step.
"""
from pathlib import Path

import numpy as np

from gausskit.gaussian.log_parser import read_normal_modes
from gausskit.utils.mode_scan import load_ts, emit_rectilinear_mode, make_qgrid
from gausskit.utils.rotor.character import _symbols


def _rot(v, axis, th):
    """Rodrigues rotation of vector v about unit `axis` by angle th (rad)."""
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(th), np.sin(th)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def _thetas(grid):
    lo, hi = grid["theta_deg"]
    return list(range(int(lo), int(hi) + 1, int(grid["dtheta"])))


def emit_scans(plan, outdir, mem="16GB", nproc=16):
    """Generate all scan inputs for a roadmap. Returns the list of written gjf paths."""
    log = plan["log"]
    # only geometry + masses are needed here (the roadmap already carries each mode's freq/axis);
    # freqs/cart are unused in emit_scans.
    numbers, pos, mass, _freqs, _cart = read_normal_modes(log)
    sym = _symbols(numbers)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    obj = modes = None                                   # lazy: only load_ts for mode-scan modes
    written = []
    for m in plan["modes"]:
        k, method = m["mode"], m["method"]
        mdir = outdir / f"mode{k}"
        if method == "rigid-rotor":
            idx = plan["fragments"][m["fragment"]]
            axis = np.asarray(m["axis"], float)
            ms = mass[idx]
            com = (ms[:, None] * pos[idx]).sum(0) / ms.sum()
            mdir.mkdir(exist_ok=True)
            for i, thd in enumerate(_thetas(m["grid"])):
                G = pos.copy()
                for a in idx:
                    G[a] = com + _rot(pos[a] - com, axis, np.radians(thd))
                lines = [f"%mem={mem}", f"%nprocshared={nproc}", plan["route"], "",
                         f"mode{k} frag{m['fragment']} rotation about COM theta={thd} deg", "",
                         plan["charge_mult"]]
                for s, v in zip(sym, G):
                    lines.append(f"{s:2s} {v[0]:18.10f} {v[1]:18.10f} {v[2]:18.10f}")
                p = mdir / f"r_{i:02d}.gjf"
                p.write_text("\n".join(lines) + "\n\n")
                written.append(str(p))
        elif method == "mode-scan":
            if obj is None:
                obj, modes, _ = load_ts(log)
            g = m["grid"]
            qgrid = make_qgrid(g["qmax"], g["dense"], g["step"])
            emitted = emit_rectilinear_mode(
                obj, modes, m["gauss_idx"], qgrid, str(mdir), freq=float(m["freq_cm"]),
                route=plan["route"], charge_mult=plan["charge_mult"], mem=mem, nproc=nproc,
                prefix=f"sm{k}", label=f"mode {k}")
            written += [str(mdir / name) for name, _ in emitted]
        # harmonic -> no scan
    return written
