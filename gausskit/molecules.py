from pathlib import Path
import numpy as np

from ase import Atoms
from ase.io.gaussian import read_gaussian_out
from ase.symbols import symbols2numbers
from gausskit.gaussian.log_parser import (
    read_external_symmetry_number,
    read_charge_and_multiplicity,
    read_log_energy,
    read_log_parameters,
)
from gausskit.gaussian.anharm import (
    read_harm_freq,
    read_anharm_matrix,
    read_harm_freq_another,
)
from gausskit.gaussian.hindrot import Hinderedrotor, read_hindrot


def method_parser(method):
    """
    symplify the method name. TO IMPROVE!
    """
    new_method = method
    if "mp2" in method.lower():
        new_method = "MP2"
    return new_method


class Molecules(Atoms):
    """
    A Molecules is a modified Object inherited from ase.Atoms
    particularly adapted to Gaussian output files
    """

    def __init__(
        self,
        *args,
        symbols=None,
        positions=None,
        numbers=None,
        tags=None,
        masses=None,
        info=None,
    ):
        super().__init__(
            symbols=symbols,
            positions=positions,
            numbers=numbers,
            tags=tags,
            masses=masses,
            info=info,
        )

        # initialize some properties
        self._charge = None
        self._multiplicity = None
        self._electronic_energy = None
        self._external_symmetry_number = None
        self._zpe = None
        self._frequencies = np.array([])
        self._anharm_zpe = None
        self._anharm_X_matrix = None
        self._method = None
        self._basis = None
        self._ts = None
        self._hinderedrotor = Hinderedrotor([], [], [], [], [], [], [])
        self.logpath = None

    @property
    def charge(self):
        """return charge"""
        if self._charge == None:
            print("charge is not set")
        return self._charge

    @charge.setter
    def charge(self, chg):
        self._charge = chg

    @property
    def multiplicity(self):
        """return multiplicity"""
        if self._multiplicity == None:
            print("multiplicity is not set")
        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, mult):
        self._multiplicity = mult

    @property
    def electronic_energy(self):
        """return electronic energy (Eele)"""
        if self._electronic_energy == None:
            print("electronic energy (Eele) is not set")
        return self._electronic_energy

    @electronic_energy.setter
    def electronic_energy(self, energy):
        self._electronic_energy = energy

    @property
    def external_symmetry_number(self):
        """return external_symmetry_number"""
        if self._external_symmetry_number == None:
            print("External Symmetry Number is not set")
        return self._external_symmetry_number

    @external_symmetry_number.setter
    def external_symmetry_number(self, ESN):
        self._external_symmetry_number = ESN

    @property
    def zpe(self):
        """return zero point energy (ZPE)"""
        if self._zpe == None:
            print("zero point energy (ZPE) is not set")
        return self._zpe

    @zpe.setter
    def zpe(self, energy):
        self._zpe = energy

    @property
    def anharm_zpe(self):
        """return anharmonic zero potential energy"""
        if self._anharm_zpe == None:
            print("anharmonic zero point energy is not set")
        return self._anharm_zpe

    @anharm_zpe.setter
    def anharm_zpe(self, energy):
        self._anharm_zpe = energy

    @property
    def anharm_matrix(self):
        """return anharmonic X matrix"""
        if type(self._anharm_X_matrix) == type(None):
            print("anharmonic X matrix is not set")
        return self._anharm_X_matrix

    @anharm_matrix.setter
    def anharm_matrix(self, matrix):
        self._anharm_X_matrix = matrix

    @property
    def frequencies(self):
        """return vibrational frequencies"""
        return self._frequencies

    @frequencies.setter
    def frequencies(self, _frequencies):
        self._frequencies = _frequencies

    @property
    def hinderedrotor(self):
        """return hindrot"""
        return self._hinderedrotor

    @hinderedrotor.setter
    def hinderedrotor(self, _hinderedrotor):
        self._hinderedrotor = _hinderedrotor

    @property
    def method(self):
        """return method, or functional"""
        if self._method == None:
            print("method is not set")
        return self._method

    @method.setter
    def method(self, _method):
        self._method = _method

    @property
    def basis(self):
        """return basis sets"""
        if self._basis == None:
            print("basis sets are not set")
        return self._basis

    def set_ts(self):
        # set if TS from frequency check
        self._ts = None
        if self.frequencies.any():
            if (self.frequencies > 0).all():
                self._ts = False
            elif len(self.frequencies) > 1:
                sorted_frequancies = self.frequencies.copy()
                sorted_frequancies.sort()
                if sorted_frequancies[0] < 0 and sorted_frequancies[1:].all() > 0:
                    self._ts = True
                else:
                    print("More than one imag freq! Please check!")
            else:
                if sorted_frequancies[0] < 0:
                    self._ts = True
        # single atom have no ts
        elif len(self.get_chemical_symbols()) == 1:
            self._ts = False
        else:
            print("No frequency (TS not possible.)")

    @property
    def ts(self):
        """return if calculation is transition state"""
        if self._ts == None:
            self.set_ts()
        return self._ts

    @classmethod
    def from_log(cls, filename):

        parameters = read_log_parameters(filename)
        if parameters == None:
            return cls()

        anharm = parameters["anharm"]
        freq = parameters["freq"]
        hindrot = parameters["hindrot"]
        method = parameters["method"]
        basis = parameters["basis"]

        energy = read_log_energy(
            filename,
            method=method,
            freq=freq,
            anharm=anharm,
        )

        with open(filename, "r", encoding="utf-8") as f:
            atoms = read_gaussian_out(f)

        symbols = atoms.symbols
        numbers = atoms.numbers
        positions = atoms.positions

        new_mol = cls(
            numbers=numbers,
            positions=positions,
        )

        if energy == None:
            return new_mol

        # set charge and multiplicity
        _charge, _mult = read_charge_and_multiplicity(filename)
        new_mol.charge = _charge
        new_mol.multiplicity = _mult

        # set external symmetry number
        ESN = read_external_symmetry_number(filename)
        new_mol.external_symmetry_number = ESN

        # set energy
        new_mol.electronic_energy = energy["Eele"]
        if freq:
            new_mol.zpe = energy["ZPE"]
            _frequencies = read_harm_freq(filename)
            if _frequencies == []:
                _frequencies = read_harm_freq_another(filename)
            new_mol.frequencies = np.array(_frequencies)
            # set transition state from frequencies
            new_mol.set_ts()
            if anharm:
                new_mol.anharm_zpe = energy["ZPE_anharm"]
                new_mol.anharm_matrix = read_anharm_matrix(filename)

        # set methods
        new_mol.method = method_parser(method)

        # set hindrotor
        if hindrot:
            hinderedrotor = read_hindrot(filename)
            new_mol.hinderedrotor = hinderedrotor

        # set log path
        new_mol.set_filepath(filename)
        return new_mol

    def set_filepath(self, filepath):
        logpath = Path(filepath).resolve()
        self.logpath = logpath
        return

    def to_dict(self):
        mol_dict = {}
        # store Atoms object info
        mol_dict["symbols"] = self.get_chemical_symbols()
        mol_dict["positions"] = self.get_positions()
        # store other attributes
        mol_dict["charge"] = self._charge
        mol_dict["multiplicity"] = self._multiplicity
        mol_dict["external_symmetry_number"] = self._external_symmetry_number
        mol_dict["electronic_energy"] = self._electronic_energy
        mol_dict["zpe"] = self._zpe
        mol_dict["frequencies"] = self._frequencies
        mol_dict["anharm_zpe"] = self._anharm_zpe
        mol_dict["anharm_X_matrix"] = self._anharm_X_matrix
        mol_dict["method"] = self._method
        mol_dict["basis"] = self._basis
        mol_dict["ts"] = self._ts
        mol_dict["hinderedrotor"] = self._hinderedrotor.to_dict()
        mol_dict["logpath"] = self.logpath

        return mol_dict

    @classmethod
    def from_dict(cls, mol_dict):
        symbols = mol_dict["symbols"]
        numbers = symbols2numbers(symbols)
        positions = mol_dict["positions"]
        new_mol = cls(numbers=numbers, positions=positions)
        new_mol._charge = mol_dict["charge"]
        new_mol._multiplicity = mol_dict["multiplicity"]
        new_mol._electronic_energy = mol_dict["electronic_energy"]
        new_mol._external_symmetry_number = mol_dict["external_symmetry_number"]
        new_mol._zpe = mol_dict["zpe"]
        new_mol._frequencies = np.asarray(mol_dict["frequencies"])
        new_mol._anharm_zpe = mol_dict["anharm_zpe"]
        new_mol._anharm_X_matrix = np.asarray(mol_dict["anharm_X_matrix"])
        new_mol._method = mol_dict["method"]
        new_mol._basis = mol_dict["basis"]
        new_mol._ts = mol_dict["ts"]
        new_mol._hinderedrotor = Hinderedrotor.from_dict(mol_dict["hinderedrotor"])
        new_mol.logpath = mol_dict["logpath"]

        return new_mol
