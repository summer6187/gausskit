# hinderedrotor analysis
import sys
import numpy as np
from ase import units

class Hinderedrotor:
    def __init__(self, 
                reduced_moms,
                rotating_bonds, 
                rotating_groups, 
                corrected_vibs,
                symmetry_numbers,
                periodicity,
                multiplicity) -> None:
        self._reduced_moms = reduced_moms
        self._rotating_bonds = rotating_bonds
        self._rotating_groups = rotating_groups
        self._corrected_vibs = corrected_vibs
        self._symmetry_numbers = symmetry_numbers
        self._periodicity = periodicity
        self._multiplicity = multiplicity
        self._rotors = []
    
    def __call__(self) -> list:
        return self._reduced_moms

    @property
    def rotors(self):
        """return a list of objects of Rotor"""
        if self._rotors == []:
            for i in range(len(self._reduced_moms)):
                this_rotor = Rotor( self._reduced_moms[i],
                                    self._rotating_bonds[i],
                                    self._rotating_groups[i],
                                    self._corrected_vibs[i],
                                    self._symmetry_numbers[i],
                                    self._periodicity[i],
                                    self._multiplicity[i],
                                    )
                self._rotors.append(this_rotor)
        return self._rotors
    
    @ rotors.setter
    def append_rotor(self, 
                    reduced_mom,
                    rotating_bond, 
                    rotating_group, 
                    corrected_vib,
                    symmetry_number,
                    periodicity,
                    multiplicity):
        this_rotor = Rotor(reduced_mom,
                        rotating_bond, 
                        rotating_group, 
                        corrected_vib,
                        symmetry_number,
                        periodicity,
                        multiplicity)
        self._rotors.append(this_rotor)

    def to_dict(self):
        hindrot_dict = {}
        hindrot_dict["reduced_moms"] = self._reduced_moms
        hindrot_dict["rotating_bonds"] = self._rotating_bonds
        hindrot_dict["rotating_groups"] = self._rotating_groups
        hindrot_dict["corrected_vibs"] = self._corrected_vibs
        hindrot_dict["symmetry_numbers"] = self._symmetry_numbers
        hindrot_dict["periodicity"] = self._periodicity
        hindrot_dict["multiplicity"] = self._multiplicity
        return hindrot_dict

    @classmethod
    def from_dict(cls, hindrot_dict):
        reduced_moms = hindrot_dict["reduced_moms"] 
        rotating_bonds = hindrot_dict["rotating_bonds"] 
        rotating_groups = hindrot_dict["rotating_groups"] 
        corrected_vibs = hindrot_dict["corrected_vibs"]
        symmetry_numbers = hindrot_dict["symmetry_numbers"]
        periodicity = hindrot_dict["periodicity"]
        multiplicity = hindrot_dict["multiplicity"] 
        return cls(reduced_moms,
                    rotating_bonds,
                    rotating_groups,
                    corrected_vibs,
                    symmetry_numbers,
                    periodicity,
                    multiplicity)

class Rotor:
    def __init__(self, 
                reduced_mom,
                rotating_bond, 
                rotating_group, 
                corrected_vib,
                symmetry_numbers,
                periodicity,
                multiplicity) -> None:
        self.reduced_mom = reduced_mom
        self.rotating_bond = rotating_bond
        self.rotating_group = rotating_group
        self.corrected_vib = corrected_vib
        self.symmetry_numbers = symmetry_numbers
        self.periodicity = periodicity
        self.multiplicity = multiplicity



def read_hindrot(filename):
    """
    This function takes a gaussian log file with hindered rotor analysis
    filename: example.log
    this function returns a Hindered rotor
    return Hinderedrotor object
    """
    with open(filename) as f:
        found_hindrot = False
        reduced_moms = []
        rotating_bonds = []
        rotating_groups = []
        rotating_bonds_check = []
        symmetry_numbers = []
        periodicity = []
        multiplicity = []
        corrected_vibs = []
        lines = f.readlines()
        for n_line,line in enumerate(lines):
            # oh such stupid code!
            if "Identification of rotating group for bond" in line:
                found_hindrot = True
                rotating_bond = [int(line.split()[-3]), int(line.split()[-1])]
                # change to Zero-based numbering
                rotating_bond = tuple([n-1 for n in rotating_bond])
                rotating_bond = np.asarray(rotating_bond, dtype=int)
                rotating_bonds.append(rotating_bond)
            if "Composition of rotating group" in line:
                rg_n_line_begin = n_line+1
                # get the ending line number for reading rotating groups
                rg_n_line_count = -1
                for _n_line, _line in enumerate(lines[rg_n_line_begin:]):
                    rg_n_line_count += 1
                    if "Geometrical symmetry number" in _line:
                        break
                rot_grp_line = "".join(lines[rg_n_line_begin:rg_n_line_begin+rg_n_line_count])
                rotating_group = rot_grp_line.split()
                rotating_group = np.asarray(rotating_group, dtype=int)
                # change to Zero-based numbering
                rotating_group = tuple([n-1 for n in rotating_group])
                rotating_groups.append(rotating_group)
            if "Reduced Moments ---" in line:
                _reduced_moms = line.split()[3:]
                reduced_moms.extend(_reduced_moms)
            if "Identified internal rotation modes" in line:
                rm_n_line_begin = n_line+1
                rm_n_line_count = -1
                # get the ending line number for reading corrected_vibs
                for _n_line, _line in enumerate(lines[rm_n_line_begin:]):
                    rm_n_line_count += 1
                    if "Vibrational temperatures" in _line:
                        break
                corr_vibs_line = "".join(lines[rm_n_line_begin:rm_n_line_begin+rm_n_line_count])
                corrected_vibs = corr_vibs_line.split()
                corrected_vibs = np.asarray(corrected_vibs, dtype=int)
                # change to Zero-based numbering
                corrected_vibs = corrected_vibs - 1
            if "Rotor          Bond     Periodicity   Symmetry Number  Multiplicity" in line:
                psm_n_line_begin = n_line+1
                psm_n_line_count = -1
                for _n_line, _line in enumerate(lines[psm_n_line_begin:]):
                    psm_n_line_count += 1
                    if "Normal Mode Analysis for Internal Rotation " in _line:
                        break
                for _line in lines[psm_n_line_begin:psm_n_line_begin+psm_n_line_count]:
                    _line_list = _line.split()
                    rotating_bonds_check.append([_line_list[-5],_line_list[-4]])
                    periodicity.append(int(_line_list[-3]))
                    symmetry_numbers.append(int(_line_list[-2]))
                    multiplicity.append(int(_line_list[-1]))

        reduced_moms = np.asarray(reduced_moms, dtype=float)
        # convert from amu*Bohr**2 to amu*Ang^2
        reduced_moms = reduced_moms*units.Bohr**2
        rotating_bonds = np.array(rotating_bonds)
        rotating_bonds_check = np.asarray(rotating_bonds_check, dtype=int) -1
        symmetry_numbers = np.array(symmetry_numbers)
        periodicity = np.array(periodicity)
        multiplicity = np.array(multiplicity)
        if not np.allclose(rotating_bonds, rotating_bonds_check):
            print("something wrong", rotating_bonds)
        # set dtypes
        if found_hindrot:
            reduced_moms = np.array(reduced_moms)
            hinderedrotor = Hinderedrotor(reduced_moms,
                                        rotating_bonds, 
                                        rotating_groups, 
                                        corrected_vibs,
                                        symmetry_numbers,
                                        periodicity,
                                        multiplicity)
            hinderedrotor.rotors
            return hinderedrotor
        else:
            print("hindered rotor not found!")
                
if __name__ == "__main__":
    filename = sys.argv[1]
    hinderedrotor = read_hindrot(filename)
    print(hinderedrotor._reduced_moms)
    print(hinderedrotor._rotating_bonds)
    print(hinderedrotor._rotating_groups)
    print(hinderedrotor._corrected_vibs)


