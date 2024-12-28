"""a wrapper for Mominert"""
from pathlib import Path
import os
import subprocess

import numpy as np

from gausskit.molecules import Molecules
from gausskit.settings import Configuration

config = Configuration()

module = "[Mominert]"

def write_mominert(
    mol:Molecules,
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

    lines.append("  0 , 0")
    lines.append("  ")

    # these comments are default output from gauss2multi
    # lines.append("  ")
    # lines.append("  Please check internal rotors")
    # lines.append("  ")

    if verbose:
        print(f"{module:10} Write to {datfile}")
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
        print(f"{module:10} Run command: {command}")
    subprocess.run(command, shell=True, capture_output=True)

    # get default output file name
    _datname = str(datfile.absolute())
    default_outfile = Path(_datname[:len(_datname)-4] + ".out")

    # make sure the output file exists
    assert default_outfile.exists(), f"{default_outfile} doesn't exists!"

    # move the default output file to targeted outfile
    if verbose:
        print(f"{module:10} Write to {outfile}")
    default_outfile.rename(outfile.absolute())

    return

def read_mominert_out(
    outfile:Path = Path("mominert.out"),
    verbose:bool = False,
):
    with open(outfile, "r") as f:
        lines = f.readlines()

    for n, line in enumerate(lines):
        if "  REDUC" in line:
            print(f"{module:10} WARNING: Reduced moment of inertia will NOT be automatically added to .vibs file!")
        if "PRINCIP" in line:
            mominert_line = lines[n+1].split()
            (Ix, Iy, Iz) = [float(mominert_line[i]) for i in (2,5,8)]
            if verbose:
                print(f"{module:10} Read reduced moment of inertia from {outfile}")
            return Ix, Iy, Iz

    print(f"{module:10} No reduced moment of inertia found from {outfile} !!!")
    return None, None, None

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
        print(f"{module:10} Calculate k-rotor and adiabatic rotors {Krot=:.4f} {ADrot=:.4f} (amu*ang^2)")

    return Krot, ADrot
