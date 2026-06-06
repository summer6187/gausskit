# harmonic and anharmonic vibrational degrees of freedom
from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path
import numpy as np

from ase.units import Bohr, Hartree
from gausskit.degrees_of_freedom import DOF

if TYPE_CHECKING:
    from gausskit.molecules import Molecules

class Harmonic(DOF):
    def __init__(
        self,
        mol: "Molecules",
        force_constants: np.ndarray | None = None,
        masses: np.ndarray | None = None,
    ):
        super().__init__(mol)

        self._force_constants = force_constants
        self._masses = masses
        self._dynamical_matrix = None
        self._frequencies = None
        self._eigenvectors = None
        self._factor = 1. # FIXME: correct this unit
        self.results = {}

    @property
    def force_constants(self) -> np.ndarray | None:
        return self._force_constants

    @force_constants.setter
    def force_constants(self, force_constants: np.ndarray | None):
        if force_constants is None:
            self._force_constants = None
            return

        self._force_constants = np.asarray(force_constants)
        fc_shape = self._force_constants.shape

        if fc_shape[0] != fc_shape[1]:
            msg = "Force constants must be square-shaped."
            raise RuntimeError(msg)

        if len(self.molecules)*3 != fc_shape[0]:
            msg = "Force constants shape disagrees with Molecules."
            msg += f"Molecules length*3: {len(self.molecules)*3} FC shape:{fc_shape}"
            raise RuntimeError(msg)

        self._set_dynamical_matrix()

    @property
    def masses(self) -> np.ndarray:
        if self._masses is None:
            self._masses = self.molecules.get_masses()
        return self._masses

    @property
    def dynamical_matrix(self) -> np.ndarray | None:
        if self._dynamical_matrix is None:
            self._set_dynamical_matrix()
        return self._dynamical_matrix

    def _set_dynamical_matrix(self):
        rminv = (self.masses ** -0.5).repeat(3)
        self._dynamical_matrix = self.force_constants * rminv[:, None] * rminv[None, :]
        self._solve()

    def _solve(self):
        # Solve eigenvalue problem to compute vibrational frequencies and eigenvectors
        w2_s, X_is = np.linalg.eigh(np.asarray(self.dynamical_matrix))

        # First six modes are translational and rotational so ignore:
        last_ignore_mode = 6

        # Check for imaginary frequencies
        w2min = w2_s[last_ignore_mode:].min()
        if w2min < 0:
            msg = f"imaginary frequencies found {w2min}."
            print(msg)

        nw = len(w2_s) - last_ignore_mode
        n_atoms = len(self.masses)

        w_s = np.sqrt(abs(w2_s[last_ignore_mode:])) * np.sign(w2_s[last_ignore_mode:])
        X_acs = X_is[:, last_ignore_mode:].reshape(n_atoms, 3, nw)

        self._frequencies = w_s * self._factor
        self._eigenvectors = X_acs

    @property
    def frequencies(self) -> np.ndarray | None:
        if self._frequencies is None:
            self._set_dynamical_matrix()
        return self._frequencies

    @property
    def eigenvectors(self) -> np.ndarray | None:
        if self._eigenvectors is None:
            self._set_dynamical_matrix()
        return self._eigenvectors

    @property
    def E0(self) -> float | None:
        return self.molecules.electronic_energy * Hartree

    @property
    def positions0(self) -> np.ndarray | None:
        return self.molecules.positions

    def calculate(self, atoms):
        assert np.allclose(atoms.numbers, self.molecules.numbers), "New structure not match the order with FC!"
        d = (atoms.get_positions() - self.positions0).flatten()
        # Harmonic energy is (1/2) d^T F d ; the 1/2 must not be dropped.
        energy = self.E0 + 0.5 * (self.force_constants @ d @ d)
        forces = -self.force_constants @ d  # force = -grad(E) = -F d
        results = {
            "energy": energy,
            "forces": forces,
        }
        self.results.update(results)
        return results

    def from_gaussian_fc(self, filename):
        fc = parse_gaussian_fc(filename)
        self.force_constants = fc

def parse_gaussian_fc(filename) -> np.ndarray | None:
    filename = Path(filename)
    if filename.suffix == ".fchk":
        fc = read_force_constants_fchk(filename)
    else:
        print(f"Force constants from {filename} not implemented.")
        fc = None
    return fc

def read_force_constants_fchk(filename, verbose=False):
    with open(filename) as f:
        lines = f.readlines()

    read = False
    for n,line in enumerate(lines):
        if read and line.split()[0][0].isalpha():
            # read only the numbers. 
            # If we meet letters str.isalpha(), then we stop read.
            l_end = n-1
            break
        if line.startswith("Cartesian Force Constants"):
            l_start = n+1
            read = True

    fc_array = " ".join(lines[l_start:l_end+1]).split()
    fc_array = np.array([float(_f) for _f in fc_array])

    # Force constants in Gaussian is stored in a lower triangle matrix.
    # Full force constants matrix is in 3*n * 3*n.
    # The lower triangle matrix has (3*n*3*n + 3*n)/2 elements.
    # For a given array lengh, for example 171, we can calculate
    # the number of atoms in the system by solving (3*n*3*n + 3*n)/2 = 171.
    # We need to solve the equation: 4.5 * n^2 + 1.5 * n - 171 = 0
    # The root is given by (-b +/- sqrt(b^2 - 4ac)) / 2a
    n_array = fc_array.shape[0]
    _n_atoms = (-1.5 + np.sqrt(1.5**2 + 4*4.5*n_array)) / 9
    n_atoms = int(_n_atoms)
    assert np.allclose(n_atoms, _n_atoms) # make sure n_atoms is int

    # restore the array into lower triangle matrix
    fc = np.zeros((3*n_atoms , 3*n_atoms))
    for i in range(3*n_atoms):
        # i th row will begin from index 1+2+...+i in fc_array
        # This row will have (i+1) elements
        _fc_array_index_1 = (1 + i) * i / 2
        fc_array_index_1 = int(_fc_array_index_1)
        assert np.allclose(fc_array_index_1, _fc_array_index_1)
        fc[i,:][0:i+1] = fc_array[fc_array_index_1:fc_array_index_1 + i + 1]
        if verbose:
            # for debug or understand the details
            print(f"{i} {fc_array_index_1 = } {fc_array_index_1 + i + 1 = }")
            print(f"{fc_array[fc_array_index_1:fc_array_index_1 + i + 1] = }")

    # recover the full fc matrix
    for i in range(3*n_atoms):
        fc[i,:] = fc[:,i]
    if verbose:
        print(f"{fc = }")

    # Gaussian FC unit is Hartree / Bohr / Bohr
    # Convert to eV / angstrom / angstrom
    fc *= Hartree / Bohr / Bohr

    return fc
