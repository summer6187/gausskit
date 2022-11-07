from ase import Atoms
from .log_parser import read_log_energy, read_log_parameters
from ase.io.gaussian import read_gaussian_out
from .get_anharm_input import read_harm_freq, read_anharm_matrix, read_harm_freq_another
import numpy as np
import os

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
        self._electronic_energy = None
        self._zpe = None
        self._frequencies = np.array([])
        self._anharm_zpe = None
        self._anharm_X_matrix = None
        self._method = None
        self._basis = None
        self._ts = None
        self.logpath = None

    @property
    def electronic_energy(self):
        """return electronic energy (Eele)"""
        if self._electronic_energy == None:
            print("electronic energy (Eele) is not set")
        return self._electronic_energy
    
    @ electronic_energy.setter
    def electronic_energy(self, energy):
        self._electronic_energy = energy

    @property
    def zpe(self):
        """return zero point energy (ZPE)"""
        if self._zpe == None:
            print("zero point energy (ZPE) is not set")
        return self._zpe

    @ zpe.setter
    def zpe(self, energy):
        self._zpe = energy

    @property
    def anharm_zpe(self):
        """return anharmonic zero potential energy"""
        if self._anharm_zpe == None:
            print("anharmonic zero point energy is not set")
        return self._anharm_zpe

    @ anharm_zpe.setter
    def anharm_zpe(self, energy):
        self._anharm_zpe = energy

    @property
    def anharm_matrix(self):
        """return anharmonic X matrix"""
        if type(self._anharm_X_matrix) == type(None):
            print("anharmonic X matrix is not set")
        return self._anharm_X_matrix
    
    @ anharm_matrix.setter
    def anharm_matrix(self, matrix):
        self._anharm_X_matrix = matrix

    @property
    def frequencies(self):
        """return vibrational frequencies"""
        return self._frequencies
    
    @ frequencies.setter
    def frequencies(self, _frequencies):
        self._frequencies = _frequencies
    
    @property
    def method(self):
        """return method, or functional"""
        if self._method == None:
            print("method is not set")
        return self._method

    @ method.setter
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
        method = parameters["method"]
        basis = parameters["basis"]

        energy = read_log_energy(filename, 
            method=method,
            freq = freq,
            anharm = anharm,
            )

        with open(filename, "r", encoding="utf-8") as f:
            atoms = read_gaussian_out(f)
        
        symbols = atoms.symbols
        numbers = atoms.numbers
        positions = atoms.positions

        new_mol = cls(
            numbers = numbers,
            positions = positions,
        )

        if energy == None:
            return  new_mol

        # set energy
        new_mol.electronic_energy = energy["Eele"]
        if freq:
            new_mol.zpe = energy["ZPE"]
            _frequencies = read_harm_freq(filename)
            if _frequencies == []:
                _frequencies  = read_harm_freq_another(filename)
            new_mol.frequencies = np.array(_frequencies)
            # set transition state from frequencies
            new_mol.set_ts()
            if anharm:
                new_mol.anharm_zpe = energy["ZPE_anharm"]
                new_mol.anharm_matrix = read_anharm_matrix(filename)
        
        # set methods
        new_mol.method = method

        # set log path
        new_mol.set_filepath(filename)
        return new_mol

    def set_filepath(self, filepath):
        logpath = os.path.abspath(filepath)
        self.logpath = logpath
        return
