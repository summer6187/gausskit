"""a wrapper for Densum"""
from pathlib import Path
import os
import subprocess

import numpy as np

from gausskit.molecules import Molecules
from gausskit.settings import Configuration

config = Configuration()

module = "[Densum]"

def get_degrees_of_freedom_lines(
    mol:Molecules,
    krot:float,
    ad_rot:float,
    if_ktools:bool = False,
):
    """Create the degree-of-freedom block for ``densum``.

    Args:
        mol (Molecules): Molecule with frequencies attached.
        krot (float): K-rotor constant in cm⁻¹.
        ad_rot (float): 2D adiabatic rotor constant in cm⁻¹.
        if_ktools (bool, optional): Use ``kro``/``jro`` labels when
            ``True``. Defaults to ``False``.

    Returns:
        list[str]: Lines to be appended to ``densum.dat``.
    """

    # get all non imaginary frequencies
    freq = mol.frequencies
    nonimg_freq = freq[freq>0]

    # degrees of freedom
    len(nonimg_freq)

    lines = []

    rottype = "vib"
    # indexing from 1; init so the K-rotor/2D-rotor blocks below still work
    # when there are no vibrations (e.g. a monatomic such as the H product).
    i = 0
    for i in range(1, len(nonimg_freq)+1):
        lines.append(f" {i:3d}   {rottype:6} {nonimg_freq[i-1]:12.4f}   0.0   1")

    if if_ktools:
        rottype = "kro"
    else:
        rottype = "qrot " if (krot < 11.0) else "rot  "
    if np.abs(krot) > 1e-12:
        i += 1
        lines.append(
            f" {i:3d}   {rottype:6} {krot:12.4f}   1.0   1   ! K-rotor"
        )

    if if_ktools:
        rottype = "jro"
    else:
        rottype = "qrot " if (ad_rot < 11.0) else "rot  "
    if np.abs(ad_rot) > 1e-12:
        i += 1
        lines.append(
            f" {i:3d}   {rottype:6} {ad_rot:12.4f}   1.0   2   ! 2D adiabatic rotor"
        )
    return lines


def write_densum(
    mol:Molecules,
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

    # k-rotor and 2D adiabatic rotor
    krot = mol.krotor
    ad_rot = mol.ad_rotor

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

    dof_lines = get_degrees_of_freedom_lines(
        mol, krot, ad_rot
    )
    lines += dof_lines

    lines.append("  ")

    if verbose:
        print(f"{module:11} Write to {datfile}")

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
    """Run the external ``densum`` program.

    Args:
        datfile (Path): Input ``densum`` control file.
        outfile (Path): Expected output file name.
        verbose (bool, optional): Print executed command. Defaults to ``False``.

    Returns:
        None
    """

    cwd = datfile.parent.absolute()

    # We do this because densum program only read densum.dat file
    if datfile.name != "densum.dat":
        default_datfile = cwd / "densum.dat"
        default_datfile.write_text(datfile.read_text())

    # run densum
    command = f"cd {cwd};" + config.machine.densum_command
    if verbose:
        print(f"{module:11} Run command: {command}")
    subprocess.run(command, shell=True, capture_output=True)

    # get default output file name
    default_outfile = cwd / "densum.out"

    # make sure the output file exists
    assert default_outfile.exists(), f"{default_outfile} doesn't exists!"

    # move the default output file to targeted outfile
    # This is not needed, fname line in densum.dat will determine the output filename fname.dens
    # default_outfile.rename(outfile.absolute())

    return

