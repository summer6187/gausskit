"""Minimal pure-Python Gaussian formatted-checkpoint (.fchk) reader.

Replaces the McUtils dependency for reading force constants and related blocks.
The fchk format is regular:

    <name, left-justified ~40 chars> <type I/R/C/L> [N=  <count>]
    <values...>            (only for arrays, on the following lines)

Scalars have the value on the header line; arrays declare "N= count" and list the
values (5 per line for reals, 6 for ints) on subsequent lines.

This reader is intentionally minimal: a generic block reader plus typed accessors
for the blocks we use. The FCHK class returns values RAW (atomic units, as stored
by Gaussian); unit conversions are the caller's responsibility (see VPT2). The
module also exposes one converting convenience, parse_gaussian_fc, which returns
the force-constant matrix in eV/Angstrom^2.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ase.units import Bohr, Hartree

_TYPES = {"I": int, "R": float, "C": str, "L": int}


class FCHK:
    def __init__(self, path):
        with open(path) as f:
            self._lines = f.readlines()

    # ---- generic block access -------------------------------------------------
    def _find_header(self, name):
        """Return (line_index, type_char, count or None) for a named block."""
        for k, line in enumerate(self._lines):
            if not line.startswith(name):
                continue
            rest = line[len(name):]
            if not rest[:1].isspace():           # name must be a full field, not a prefix
                continue
            toks = rest.split()
            if not toks or toks[0] not in _TYPES:
                continue
            tchar = toks[0]
            count = int(toks[2]) if len(toks) >= 3 and toks[1] == "N=" else None
            return k, tchar, count
        raise KeyError(f"fchk block not found: {name!r}")

    def scalar(self, name):
        k, tchar, count = self._find_header(name)
        if count is not None:
            raise ValueError(f"{name!r} is an array, not a scalar")
        return _TYPES[tchar](self._lines[k].split()[-1])

    def array(self, name):
        """Return a 1-D np.ndarray for an array block (raw units)."""
        k, tchar, count = self._find_header(name)
        if count is None:
            raise ValueError(f"{name!r} is a scalar, not an array")
        conv = _TYPES[tchar]
        vals, j = [], k + 1
        while len(vals) < count:
            vals.extend(conv(tok) for tok in self._lines[j].split())
            j += 1
        return np.array(vals[:count])

    # ---- typed convenience accessors -----------------------------------------
    @property
    def n_atoms(self):
        return self.scalar("Number of atoms")

    @property
    def n_modes(self):
        return self.scalar("Number of Normal Modes")

    @property
    def atomic_numbers(self):
        return self.array("Atomic numbers").astype(int)

    @property
    def masses(self):
        """Atomic masses (amu), from 'Real atomic weights'."""
        return self.array("Real atomic weights")

    @property
    def coordinates(self):
        """Geometry as (n_atoms, 3) in BOHR (raw 'Current cartesian coordinates')."""
        return self.array("Current cartesian coordinates").reshape(-1, 3)

    @property
    def hessian(self):
        """Cartesian force constants (3N,3N), symmetric, in Hartree/Bohr^2."""
        n3 = 3 * self.n_atoms
        tri = self.array("Cartesian Force Constants")
        H = np.zeros((n3, n3))
        H[np.tril_indices(n3)] = tri
        return H + np.tril(H, -1).T

    @property
    def vib_modes(self):
        """Gaussian normal modes (n_modes, 3N), Cartesian, Gaussian order & sign."""
        return self.array("Vib-Modes").reshape(self.n_modes, 3 * self.n_atoms)

    def _force_derivatives(self):
        """Split 'Cartesian 3rd/4th derivatives' into (cubic, quartic) tensors
        (n_modes, 3N, 3N), symmetric in the two Cartesian legs; the mode leg is
        in Gaussian's Vib-Modes order. Raw atomic units."""
        n3 = 3 * self.n_atoms
        m = self.n_modes
        vals = self.array("Cartesian 3rd/4th derivatives")
        half = len(vals) // 2
        ti = np.tril_indices(n3)
        out = []
        for chunk in (vals[:half], vals[half:]):
            per = chunk.reshape(m, len(ti[0]))
            T = np.zeros((m, n3, n3))
            for i in range(m):
                T[i][ti] = per[i]
                T[i][ti[1], ti[0]] = per[i]
            out.append(T)
        return out[0], out[1]

    @property
    def cubic(self):
        """d^3 V / dQ_i dx_A dx_B  (n_modes, 3N, 3N), raw a.u."""
        return self._force_derivatives()[0]

    @property
    def quartic(self):
        """d^4 V / dQ_i^2 dx_A dx_B  (n_modes, 3N, 3N), semi-diagonal, raw a.u."""
        return self._force_derivatives()[1]


def parse_gaussian_fc(filename) -> np.ndarray | None:
    """Cartesian force constants (3N, 3N) in eV/Angstrom^2 from a Gaussian .fchk.

    Convenience converter (unlike the raw `FCHK.hessian`): reads the lower-triangular
    'Cartesian Force Constants' block and converts Hartree/Bohr^2 -> eV/Angstrom^2.
    Returns None for unsupported file types.
    """
    filename = Path(filename)
    if filename.suffix == ".fchk":
        return FCHK(filename).hessian * (Hartree / Bohr / Bohr)
    print(f"Force constants from {filename} not implemented.")
    return None


if __name__ == "__main__":
    import sys
    f = FCHK(sys.argv[1])
    print("n_atoms", f.n_atoms, "n_modes", f.n_modes)
    print("Z", f.atomic_numbers, "masses", np.round(f.masses, 3))
    print("hessian", f.hessian.shape, "vib_modes", f.vib_modes.shape,
          "cubic", f.cubic.shape, "quartic", f.quartic.shape)
