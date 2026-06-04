"""External rotational symmetry number from molecular geometry.

`external_symmetry_number(atoms)` computes the external rotational symmetry
number sigma of a molecule directly from its geometry, using the vendored,
trimmed pymatgen point-group analyzer in ``_pointgroup.py``. This is the
ground truth regardless of how Gaussian was run; in particular it is correct
under ``nosymm`` (where Gaussian reports sigma=1 for every species).

A thin ``ase.Atoms`` -> duck-typed-molecule adapter (``_Species`` / ``_Site`` /
``_Molecule``) feeds the analyzer; only the interface PointGroupAnalyzer touches
is implemented.
"""

from __future__ import annotations

import warnings

import numpy as np

from gausskit.symmetry._pointgroup import PointGroupAnalyzer

# Tolerances validated against the CH2O+OH entrance scan (reproduces the paper's
# C2v/Cs boundary exactly). See the _pointgroup.py module docstring for why the
# eigenvalue tolerance is tightened from pymatgen's 0.01 default.
DEFAULT_TOLERANCE = 0.2
DEFAULT_EIGEN_TOLERANCE = 1e-4


class _Species:
    """Minimal stand-in for a pymatgen Species: an element label with a mass.

    Equality/hashing are by element symbol only (isotope-independent), as the
    point-group analyzer compares species to decide if two sites are equivalent.
    """

    __slots__ = ("symbol", "weight")

    def __init__(self, symbol: str, weight: float) -> None:
        self.symbol = symbol
        self.weight = weight

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Species) and other.symbol == self.symbol

    def __hash__(self) -> int:
        return hash(self.symbol)

    def __repr__(self) -> str:
        return self.symbol


class _Site:
    """A site = a species at a Cartesian coordinate."""

    __slots__ = ("species", "coords")

    def __init__(self, species: _Species, coords) -> None:
        self.species = species
        self.coords = np.asarray(coords, dtype=float)


class _Molecule:
    """Duck-typed molecule providing exactly the interface PointGroupAnalyzer uses."""

    def __init__(self, sites: list[_Site]) -> None:
        self._sites = sites

    def __len__(self) -> int:
        return len(self._sites)

    def __iter__(self):
        return iter(self._sites)

    def __getitem__(self, idx: int) -> _Site:
        return self._sites[idx]

    @property
    def cart_coords(self) -> np.ndarray:
        return np.array([s.coords for s in self._sites])

    def get_centered_molecule(self) -> _Molecule:
        """Return a copy translated so the center of mass is at the origin."""
        masses = np.array([s.species.weight for s in self._sites])
        coords = self.cart_coords
        com = (masses[:, None] * coords).sum(axis=0) / masses.sum()
        return _Molecule([_Site(s.species, s.coords - com) for s in self._sites])

    @classmethod
    def from_ase(cls, atoms) -> _Molecule:
        symbols = atoms.get_chemical_symbols()
        masses = atoms.get_masses()
        coords = atoms.get_positions()
        return cls([_Species_site(sym, m, c) for sym, m, c in zip(symbols, masses, coords)])


def _Species_site(symbol: str, mass: float, coords) -> _Site:
    return _Site(_Species(symbol, float(mass)), coords)


def external_symmetry_number(
    atoms,
    tolerance: float = DEFAULT_TOLERANCE,
    eigen_tolerance: float = DEFAULT_EIGEN_TOLERANCE,
):
    """External rotational symmetry number of an ``ase.Atoms`` from its geometry.

    Args:
        atoms: An ``ase.Atoms`` (or subclass, e.g. gausskit ``Molecules``) carrying
            positions and chemical symbols.
        tolerance: Distance tolerance (Angstrom) for mapping a rotated atom onto an
            equivalent atom. Defaults to 0.2.
        eigen_tolerance: Tolerance on the normalized inertia-tensor eigenvalues used
            to classify the top (linear / spherical / symmetric / asymmetric).
            Defaults to 1e-4 (see module docstring).

    Returns:
        The integer symmetry number, or ``None`` if it could not be determined
        (no geometry, or the analyzer raised) so the caller can fall back.
    """
    try:
        n = len(atoms)
    except TypeError:
        return None
    if n == 0:
        return None
    if n == 1:
        return 1
    try:
        mol = _Molecule.from_ase(atoms)
        pga = PointGroupAnalyzer(mol, tolerance=tolerance, eigen_tolerance=eigen_tolerance)
        return int(pga.get_rotational_symmetry_number())
    except Exception as exc:  # noqa: BLE001 - any failure -> caller falls back
        warnings.warn(
            f"external_symmetry_number: could not determine sigma from structure "
            f"({type(exc).__name__}: {exc}); caller should fall back.",
            stacklevel=2,
        )
        return None


__all__ = ["external_symmetry_number", "PointGroupAnalyzer"]
