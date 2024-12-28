"""a wrapper for Mominert"""
from pathlib import Path
import os
import subprocess

from gausskit.molecules import Molecules
from gausskit.settings import Configuration

config = Configuration()


def write_mominert(
    mol:Molecules,
    outfile:Path = Path("mominert.dat")
):
    """
    Write input file for Mominert program
    """
    lines = []
    lines.append(f" {mol.name}")
    lines.append(" ANGS")
    lines.append(f"  {len(mol.numbers)}")
    for i in range(len(mol.numbers)):
        line = f"  {mol.symbols[i]:2}    {i+1}    "
        pos = mol.positions[i]
        line += "   ".join([f"{p:8f}" for p in pos])
        lines.append(line)

    lines.append("  0 , 0")
    lines.append("  ")
    lines.append("  ")
    lines.append("  Please check internal rotors")
    lines.append("  ")

    with open(outfile, "w") as f:
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
    print(f" Run command {command}")
    subprocess.run(command, shell=True, capture_output=True)

    # get default output file name
    _datname = str(datfile.absolute())
    default_outfile = Path(_datname[:len(_datname)-4] + ".out")

    # make sure the output file exists
    assert default_outfile.exists(), f"{default_outfile} doesn't exists!"

    # move the default output file to targeted outfile
    default_outfile.rename(outfile.absolute())

    if verbose:
        print(f"[Mominert]  run command: {command}")
    return


