"""a wrapper for Densum"""
from pathlib import Path
import os
import subprocess

import numpy as np

from gausskit.molecules import Molecules
from gausskit.settings import Configuration

config = Configuration()

module = "[Densum]"

def write_densum(
    mol:Molecules,
    krot:float,
    ad_rot:float,
    fname:str = None,
    Egrain="10   3000   4000   50000",
    datfile:Path = Path("densum.dat"),
    verbose:bool = False,
):
    """
    Write input file for densum program
    """

    # get all non imaginary frequencies
    freq = mol.frequencies
    nonimg_freq = freq[freq>0]

    # degrees of freedom
    degrees_of_freedom = len(nonimg_freq)

    # densum will create fname.dens as output besides densum.out
    if fname is None:
        fname = mol.name

    lines = []
    lines.append(f" {mol.name}")
    lines.append(f" {fname}")
    if np.abs(krot) > 1e-12:
        degrees_of_freedom += 1
    lines.append(f"  {degrees_of_freedom}  0   HAR   AMUA")
    lines.append(str(Egrain))
    rottype = "vib"
    for i in range(len(nonimg_freq)):
        lines.append(f" {i+1:3d}   {rottype:6} {nonimg_freq[i]:12.4f}   0.0   1")

    rottype = "qrot " if (krot < 11.0) else "rot  "
    i = degrees_of_freedom
    if np.abs(krot) > 1e-12:
        lines.append(
            f" {i:3d}   {rottype:6} {krot:12.4f}   1.0   1   ! K-rotor"
        )

    rottype = "qrot " if (ad_rot < 11.0) else "rot  "
    if np.abs(ad_rot) > 1e-12:
        lines.append(
            f" {i+1:3d}   {rottype:6} {ad_rot:12.4f}   1.0   2   ! 2D adiabatic rotor"
        )

    lines.append("  ")

    if verbose:
        print(f"{module:10} Write to {datfile}")

    with open(datfile, "w") as f:
        for line in lines:
            f.write(f"{line} {os.linesep}")
        f.write(f"  {os.linesep}")

    return

def run_densum(
    datfile:Path = Path("densum.dat"),
    outfile:Path = Path("densum.out"),
    verbose:bool = False,
):
    cwd = datfile.parent.absolute()

    if datfile.name != "densum.dat":
        default_datfile = cwd / "densum.dat"
        default_datfile.write_text(datfile.read_text())

    # run densum
    command = f"cd {cwd};" + config.machine.densum_command
    if verbose:
        print(f"{module:10} Run command: {command}")
    subprocess.run(command, shell=True, capture_output=True)

    # get default output file name
    default_outfile = cwd / "densum.out"

    # make sure the output file exists
    assert default_outfile.exists(), f"{default_outfile} doesn't exists!"

    # move the default output file to targeted outfile
    # default_outfile.rename(outfile.absolute())

    return

