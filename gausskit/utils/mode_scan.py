"""Generate rigid normal-mode scan STRUCTURES for a TS's softest modes.

A thin convenience/demo on top of the gausskit objects: it loads a transition state
(`VPT2` from a `.fchk`, or a `Harmonic` from a freq `.log` via
`gausskit.gaussian.log_parser.read_normal_modes`), picks the lowest real modes, and uses
`Harmonic.displace_along_mode` to build the geometry at each scan point

    x(Q) = x0 + (L_i / sqrt(m)) * Q ,   Q in A*amu^(1/2),

writing one Gaussian input (.gjf) per point plus a manifest. It writes input structures
only -- you submit the jobs however you like. This is the engine behind
`gausskit utils mode_scan`.
"""
from __future__ import annotations
import os
import numpy as np

#: dense bottom (|Q|<=3 @0.25) + wide wings to |Q|=11 -- 41 points
DEFAULT_QGRID = np.unique(np.concatenate([
    np.arange(-3.0, 3.0001, 0.25),
    np.array([-11, -10, -9, -8, -7, -6, -5, -4, 4, 5, 6, 7, 8, 9, 10, 11], float)]))
DEFAULT_ROUTE = "#p m062x def2tzvp int=superfine scf=(conver=10,xqc) nosymm"


def make_qgrid(qmax=11.0, dense=3.0, step=0.25):
    """Dense window (+-dense @ step) plus integer wings out to +-qmax."""
    dense = min(dense, qmax)                                    # keep the grid within +-qmax
    wings = np.arange(np.ceil(dense) + 1, qmax + 1e-9)
    wings = np.concatenate([-wings[::-1], wings])
    return np.unique(np.concatenate([np.arange(-dense, dense + 1e-9, step), wings]))


def load_ts(structure, log=None):
    """Return ``(obj, modes, freqs)`` for an fchk or a freq log, where `obj` exposes
    `displace_along_mode(i, Q, modes=...)` (a gausskit `Harmonic`/`VPT2`):

      - `.fchk`  -> `VPT2.from_fchk` (modes=None: it uses Gaussian's normal-mode basis).
      - `.log`   -> a `Harmonic` built from the parsed geometry; `modes` is the parsed,
                    mass-weighted, normalized normal-mode matrix (nmodes, 3N).
    """
    structure = str(structure)
    if structure.lower().endswith((".fchk", ".fch")):
        from gausskit.degrees_of_freedom.vpt2 import VPT2
        v = VPT2.from_fchk(structure, log)
        freqs = np.array([float(v.frequency(i)) for i in range(v.n_modes)])
        return v, None, freqs

    from gausskit.gaussian.log_parser import read_normal_modes
    from gausskit.molecules import Molecules
    from gausskit.degrees_of_freedom.harmonic import Harmonic
    numbers, pos, mass, freqs, cart = read_normal_modes(structure)
    mol = Molecules(numbers=numbers, positions=pos)
    harm = Harmonic(mol, masses=mass)
    mw = np.repeat(mass, 3) ** 0.5
    Lg = np.array([(c.reshape(-1) * mw) for c in cart])        # mass-weight the Cartesian modes
    Lg = Lg / np.linalg.norm(Lg, axis=1, keepdims=True)        # normalize -> (nmodes, 3N)
    return harm, Lg, np.asarray(freqs)


def emit_rectilinear_mode(obj, modes, idx, qgrid, mdir, *, freq, route,
                          charge_mult="0 2", mem="16GB", nproc=16, prefix="sm", label=None):
    """Write the rigid rectilinear scan x(Q)=x0+(L/sqrt(m))Q of a SINGLE Gaussian mode `idx`
    into directory `mdir` (one gjf per Q point, title '... Q=+X.XXX'). Returns [(filename, Q)].

    Factored out of `generate_mode_scan` so a caller (e.g. `gausskit utils rotor scan`) can drive
    one Gaussian index into an arbitrary directory; the title Q-convention is the one
    `extract_scan.py` / `rotor.readback.read_rectilinear` parse back.
    """
    os.makedirs(mdir, exist_ok=True)
    label = os.path.basename(mdir.rstrip("/")) if label is None else label
    out = []
    for k, Q in enumerate(np.asarray(qgrid, float)):
        atoms = obj.displace_along_mode(int(idx), float(Q), modes=modes)      # gausskit object
        name = f"{prefix}_{k:02d}.gjf"
        _write_gjf(os.path.join(mdir, name), atoms, route, charge_mult, mem, nproc,
                   f"{label} ({freq:.1f} cm-1) Q={Q:+.3f}")
        out.append((name, float(Q)))
    return out


def generate_mode_scan(structure, log=None, n_modes=2, qgrid=None, route=DEFAULT_ROUTE,
                       charge_mult="0 2", mem="16GB", nproc=16, outdir="mode_scan"):
    """Write the rigid-scan input structures along the `n_modes` lowest REAL modes of `structure`
    (a `.fchk` or freq `.log`). Returns ``(written, picks, freqs)``: the relative gjf paths, the
    chosen mode indices, and their frequencies. Input structures only -- no job submission.
    """
    qgrid = DEFAULT_QGRID if qgrid is None else np.asarray(qgrid, float)
    obj, modes, freqs = load_ts(structure, log)
    real = np.where(freqs > 0)[0]
    picks = real[np.argsort(freqs[real])[:n_modes]]
    os.makedirs(outdir, exist_ok=True)
    manifest = ["mode,freq_cm1,idx,Q,gjf"]
    written = []
    for mi, idx in enumerate(picks, start=1):
        mdir = os.path.join(outdir, f"mode{mi}")
        emitted = emit_rectilinear_mode(obj, modes, int(idx), qgrid, mdir, freq=float(freqs[idx]),
                                        route=route, charge_mult=charge_mult, mem=mem, nproc=nproc,
                                        prefix=f"sm{mi}", label=f"mode {mi}")
        for k, (name, Q) in enumerate(emitted):
            written.append(os.path.join(f"mode{mi}", name))
            manifest.append(f"{mi},{freqs[idx]:.2f},{k},{Q:+.3f},mode{mi}/{name}")
    with open(os.path.join(outdir, "manifest.csv"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    return written, list(map(int, picks)), freqs[picks]


def _write_gjf(path, atoms, route, charge_mult, mem, nproc, title):
    sym = atoms.get_chemical_symbols()
    R = np.asarray(atoms.get_positions())
    with open(path, "w") as f:
        f.write(f"%mem={mem}\n%nprocshared={nproc}\n{route}\n\n{title}\n\n{charge_mult}\n")
        for s, r in zip(sym, R):
            f.write(f"{s:2s} {r[0]:18.10f} {r[1]:18.10f} {r[2]:18.10f}\n")
        f.write("\n")
