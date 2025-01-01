"""a wrapper for Mominert"""
from pathlib import Path
import os
import subprocess

import numpy as np

from gausskit.molecules import Molecules
from gausskit.gaussian.hindrot import Hinderedrotor
from gausskit.settings import Configuration

config = Configuration()

module = "[Mominert]"

def write_mominert(
    mol:Molecules,
    hinderedrotor:Hinderedrotor = None,
    datfile:Path = Path("mominert.dat"),
    verbose:bool = False,
):
    """
    Write input file for Mominert program
    """
    lines = []
    lines.append(f" Gausskit generated: {mol.name}")
    lines.append(" ANGS")
    lines.append(f"  {len(mol.numbers)}")
    for i in range(len(mol.numbers)):
        line = f"  {mol.symbols[i]:2}    {i+1}    "
        pos = mol.positions[i]
        line += "   ".join([f"{p:8f}" for p in pos])
        lines.append(line)

    if hinderedrotor is not None:
        # internal rotor coords information
        hindrot_lines = [""]
        for i_rotor in range(len(hinderedrotor._rotating_bonds)):
            hindrot_lines.append(
                ", ".join(
                    [str(j + 1) for j in hinderedrotor._rotating_bonds[i_rotor]]
                )
            )
            hindrot_lines.append(
                str(len(hinderedrotor._rotating_groups[i_rotor]))
            )
            hindrot_lines.append(
                ", ".join(
                    [str(j + 1) for j in hinderedrotor._rotating_groups[i_rotor]]
                )
            )
            hindrot_lines.append("")
        hindrot_lines = [j + "\n" for j in internal_rotor_coords]

        # insert rotor coords information to the file
        lines += hindrot_lines

    # this is finishing lines
    lines.append("  0 , 0")
    lines.append("  ")

    # these comments are default output from gauss2multi
    # lines.append("  ")
    # lines.append("  Please check internal rotors")
    # lines.append("  ")

    if verbose:
        print(f"{module:11} Write to {datfile}")
    with open(datfile, "w") as f:
        for line in lines:
            f.write(f"{line} {os.linesep}")
        f.write(f"  {os.linesep}")

    return


def run_mominert(
    datfile:Path = Path("mominert.dat"),
    outfile:Path = Path("mominert.out"),
    verbose:bool = False,
):
    cwd = datfile.parent.absolute()

    # run mominert
    command = f"cd {cwd};" + config.machine.mominert_command + f" {datfile.name}"
    if verbose:
        print(f"{module:11} Run command: {command}")
    subprocess.run(command, shell=True, capture_output=True)

    # get default output file name
    _datname = str(datfile.absolute())
    default_outfile = Path(_datname[:len(_datname)-4] + ".out")

    # make sure the output file exists
    assert default_outfile.exists(), f"{default_outfile} doesn't exists!"

    # move the default output file to targeted outfile
    if verbose:
        print(f"{module:11} Write to {outfile}")
    default_outfile.rename(outfile.absolute())

    return


def read_mominert_out(
    outfile:Path = Path("mominert.out"),
    verbose:bool = False,
):
    with open(outfile, "r") as f:
        lines = f.readlines()
    reduced_moment_of_inertia = []
    for n, line in enumerate(lines):
        if "REDUCED MOMENT OF INERTIA" in line:
            print(f"{module:11} WARNING: Reduced moment of inertia will NOT be automatically added to .vibs file!")
            reduced_moment_of_inertia.append(float(line.split(":")[1].split()[0]))

        if "PRINCIP" in line:
            mominert_line = lines[n+1].split()
            (Ix, Iy, Iz) = [float(mominert_line[i]) for i in (2,5,8)]
            if verbose:
                print(f"{module:11} Read reduced moment of inertia from {outfile}")
            return Ix, Iy, Iz, reduced_moment_of_inertia

    print(f"{module:11} No reduced moment of inertia found from {outfile} !!!")
    return None, None, None, reduced_moment_of_inertia

def calc_rotor(Ix, Iy, Iz, verbose:bool=False):
    """
    Identifies Krot and calculates ADrot based on the
    rotational constants Ix, Iy, Iz.
    """
    fxy = np.abs(Ix - Iy) / (Ix + Iy)
    fxz = np.abs(Ix - Iz) / (Ix + Iz)
    fyz = np.abs(Iy - Iz) / (Iy + Iz)

    # Identify the smallest of the three differences
    if fxy < fxz and fxy < fyz:
        Krot = Iz
        ADrot = np.sqrt(Ix * Iy)
    elif fxz < fxy and fxz < fyz:
        Krot = Iy
        ADrot = np.sqrt(Ix * Iz)
    elif fyz < fxy and fyz < fxz:
        Krot = Ix
        ADrot = np.sqrt(Iy * Iz)
    else:
        # Symmetric top fallback
        Krot = Ix
        ADrot = np.sqrt(Iy * Iz)

    if verbose:
        print(f"{module:11} Calculate k-rotor and adiabatic rotors {Krot=:.4f} {ADrot=:.4f} (amu*ang^2)")

    return Krot, ADrot

def get_rotor(
    outfile:Path = Path("mominert.out"),
    verbose:bool = False,
):
    Ix, Iy, Iz, _ = read_mominert_out(outfile, verbose)
    Krot, ADrot = calc_rotor(Ix, Iy, Iz, verbose)
    return Krot, ADrot

