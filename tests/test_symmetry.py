"""Regression tests for structure-based external rotational symmetry numbers.

Covers every top class the vendored point-group analyzer must handle:
  - linear C*v (OH) and D*h (N2),
  - asymmetric-top C2v (H2O, H2CO) and Cs,
  - symmetric-top C3v (NH3),
  - spherical-top Td (CH4),
  - dihedral D6h (benzene),
  - the elongated CH2O...HO entrance complex (C2v) that trips pymatgen's default
    eigen_tolerance and must still resolve to sigma=2 with our pinned 1e-4.

Self-contained: standard geometries come from ase.build.molecule; the elongated
complex is inlined (one O...H = 4.0 A scan point from the CH2O+OH campaign). Runs
under pytest, or directly as `python test_symmetry.py`.
"""

from __future__ import annotations

from ase import Atoms
from ase.build import molecule

from gausskit.symmetry import external_symmetry_number as esn

# An elongated C2v entrance complex (CH2O + OH on the C=O axis, O...H = 4.0 A).
# Its tiny long-axis moment makes pymatgen's default eigen_tolerance=0.01 call it
# linear (sigma=1); the pinned 1e-4 keeps it C2v -> sigma=2.
_ELONGATED_C2V = Atoms(
    numbers=[6, 8, 1, 1, 8, 1],
    positions=[
        (-0.466710, 0.000000, -0.269406),
        (-1.495976, 0.000002, -0.862821),
        (-0.430758, -0.000002, 0.825263),
        (0.498651, 0.000000, -0.786776),
        (-5.793321, -0.000001, -3.345124),
        (-4.959756, 0.000001, -2.863387),
    ],
)

CASES = [
    ("OH (C*v)", molecule("OH"), 1),
    ("N2 (D*h)", Atoms("N2", positions=[(0, 0, 0), (0, 0, 1.10)]), 2),
    ("H2O (C2v)", molecule("H2O"), 2),
    ("H2CO (C2v)", molecule("H2CO"), 2),
    ("NH3 (C3v)", molecule("NH3"), 3),
    ("CH4 (Td)", molecule("CH4"), 12),
    ("C6H6 (D6h)", molecule("C6H6"), 12),
    ("single atom", Atoms("Ar", positions=[(0, 0, 0)]), 1),
    ("elongated C2v complex", _ELONGATED_C2V, 2),
]


def test_external_symmetry_numbers():
    for label, atoms, expected in CASES:
        got = esn(atoms)
        assert got == expected, f"{label}: expected sigma={expected}, got {got}"


def test_cs_returns_one():
    """A bent, mirror-only complex (Cs) has sigma=1."""
    cs = Atoms(
        numbers=[8, 1, 8],  # bent O...H-O, no C2
        positions=[(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (1.80, 1.40, 0.0)],
    )
    assert esn(cs) == 1


if __name__ == "__main__":
    ok = True
    for label, atoms, expected in CASES:
        got = esn(atoms)
        flag = "PASS" if got == expected else "FAIL"
        ok = ok and got == expected
        print(f"  [{flag}] {label:24s} sigma={got} (expected {expected})")
    print("ALL PASS" if ok else "SOME FAILED")
