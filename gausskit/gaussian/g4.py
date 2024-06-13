"""
read G4 file, get energy
"""
import sys
from .log_parser import *


def get_g4_energy(result_blocks):

    energy_dict = {
        "G4": get_energy(result_blocks, "G4", "", "G4"),
    }

    E0 = energy_dict["G4"]

    return E0


def read_g4_energy(filename, verbose=False):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

        # check if the calculation terminate normally
        # only check last lines is necessary
        check_normal_termination(lines[-10:])

        # get thermal data with scaled ZPE
        ZPE_scale = 0.9854
        thermal_data = parse_freq(lines, ZPE_scale, anharm)

        # read result blocks to calculate G4 Eele
        result_blocks = get_result_blocks(lines)
        E0 = get_g4_energy(result_blocks[-1:])

        Eele = E0 - thermal_data["E_ZPE"]

        # if verbose, print the result in a formated way
        if verbose:
            verbose_print(Eele, thermal_data)

        # generate energy summary
        energy = {
            "Eele": Eele,
            "E_ZPE": thermal_data["E_ZPE"],
            "E0": E0,
        }
        return energy


if __name__ == "__main__":
    filename = sys.argv[1]
    g4_energy = read_g4_energy(filename, verbose=False)
    print(g4_energy)
