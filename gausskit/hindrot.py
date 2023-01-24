# hinderedrotor analysis
import numpy as np
import sys

class Hinderedrotor:
    def __init__(self, 
                reduced_moms,
                rotating_bonds, 
                rotating_groups, 
                corrected_vibs) -> None:
        self._reduced_moms = reduced_moms
        self._rotating_bonds = rotating_bonds
        self._rotating_groups = rotating_groups
        self._corrected_vibs = corrected_vibs
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
                                    self._corrected_vibs[i]
                                    )
                self._rotors.append(this_rotor)
        return self._rotors
    
    @ rotors.setter
    def append_rotor(self, 
                    reduced_mom,
                    rotating_bond, 
                    rotating_group, 
                    corrected_vib):
        this_rotor = Rotor(reduced_mom,
                        rotating_bond, 
                        rotating_group, 
                        corrected_vib)
        self._rotors.append(this_rotor)


class Rotor:
    def __init__(self, 
                reduced_mom,
                rotating_bond, 
                rotating_group, 
                corrected_vib) -> None:
        self.reduced_mom = reduced_mom
        self.rotating_bond = rotating_bond
        self.rotating_group = rotating_group
        self.corrected_vib = corrected_vib


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
        corrected_vibs = []
        lines = f.readlines()
        for n_line,line in enumerate(lines):
            # oh such stupid code!
            if "Identification of rotating group for bond" in line:
                found_hindrot = True
                rotating_bond = [int(line.split()[-3]), int(line.split()[-1])]
                rotating_bonds.append(rotating_bond)
            if "Composition of rotating group" in line:
                rotating_group = lines[n_line+1].split()
                rotating_group = np.asarray(rotating_group, dtype=int)
                rotating_groups.append(rotating_group)
            if "Reduced Moments ---" in line:
                reduced_moms = line.split()[-3:]
                reduced_moms = np.asarray(reduced_moms, dtype=float)
            if "Identified internal rotation modes" in line:
                corrected_vibs = lines[n_line+1].split()
                corrected_vibs = np.asarray(corrected_vibs, dtype=int)
                
        # set dtypes
        if found_hindrot:
            reduced_moms = np.array(reduced_moms)

            hinderedrotor = Hinderedrotor(reduced_moms,
                                        rotating_bonds, 
                                        rotating_groups, 
                                        corrected_vibs)
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


