"""a wrapper for Parsctst"""
from pathlib import Path
import os
import subprocess
import collections

import numpy as np

from gausskit.molecules import Molecules
from gausskit.settings import Configuration
from gausskit.gaussian.anharm import format_freq_matrix

config = Configuration()

module_dict = {
    "p": "[Parsctst]",
    "d": "[Bdens]",
}
module = collections.namedtuple("module", module_dict.keys())(**module_dict)

def write_parsctst(
    mol:Molecules,
    fname:str = None,
    barrier:list[float]=[],
    Egrain="10   3000   4000   50000",
    datfile:Path = Path("parsctst.dat"),
    verbose:bool = False,
):
    harm_freq = mol.frequencies
    anharm_matrix = mol.anharm_matrix

    img_freq = 0
    img_index = None
    ind_array = np.argsort(harm_freq)
    if harm_freq[ind_array][0] < 0:
        if np.any(harm_freq[ind_array][1:] < 0):
            print("More than one imagine freq found!!!")
            print("Please check the calculation!!!")
        else:
            img_index = ind_array[0]
            img_freq = harm_freq[img_index]
    img_nn = anharm_matrix[img_index][img_index]

    # make full anharm matrix (symmetry matrix)
    # full_anharm_matrix = (
    #     anharm_matrix + anharm_matrix.T - np.diag(np.diag(anharm_matrix))
    # )
    full_anharm_matrix = anharm_matrix  # we already have full matrix
    img_anharm_array = full_anharm_matrix[img_index]
    img_anharm_array = np.delete(img_anharm_array, img_index)

    # remake the harm_freq and anharm_matrix for sctst.dat
    harm_freq = np.delete(harm_freq, img_index)
    full_anharm_matrix = np.delete(full_anharm_matrix, img_index, axis=0)
    full_anharm_matrix = np.delete(full_anharm_matrix, img_index, axis=1)
    anharm_matrix = np.tril(full_anharm_matrix)

    # prepare inputfile
    lines = []
    lines.append(fname)
    lines.append(f"At {mol.method} ?? level of theory")
    lines.append(f"Anharmonicity from {mol.method} ?? level")
    lines.append(" ")

    lines.append(f'{len(harm_freq)}, {0}, {0}, "We" ')
    lines.append(" ")
    formated_lines = format_freq_matrix(harm_freq, anharm_matrix)
    lines += formated_lines
    lines.append(" ")
    lines.append("0    'AMUA'")
    lines.append(f"{Egrain}")
    lines.append(f"'nochekstart'  {fname}.chk")
    lines.append("VPT4A")
    if len(barrier) == 0:
        barrier_text = "<forward_barrier>  <backword_barrier>"
    else:
        barrier_text = f"{barrier[0]:.4f}  {barrier[1]:.4f}"
    lines.append(f'{barrier_text}  "kcal"')
    lines.append(f"{img_freq}  {img_nn:.5E}")
    lines += [f"{item:.5E}" for item in img_anharm_array]
    lines.append(" ")
    pardata = [
        "1       !nwalkers",
        "70.d0   !perc_wind_overlap",
        "0.60d0  !flatness",
        "1       !Writing enable (1) or disable (2)",
        "0       !Seed modifier",
        "cost    !Windows balance  (cost / low / high)",
    ]
    lines += pardata
    lines.append(" ")

    # write file
    if verbose:
        print(f"{module.p:11} Writing to {datfile}")
    with open(datfile, "w") as f:
        f.writelines([line + "\n" for line in lines])

    return lines

def write_bdens(
    mol:Molecules,
    fname:str = None,
    Egrain="10   3000   4000   50000",
    datfile:Path = Path("parsctst.dat"),
    verbose:bool = False,
):
    harm_freq = mol.frequencies
    anharm_matrix = mol.anharm_matrix

    lines = []
    lines.append(fname)
    lines.append(f"At {mol.method} ?? level of theory")
    lines.append(f"Anharmonicity from {mol.method} level")
    lines.append(" ")

    lines.append(f'{len(harm_freq)}, {0}, {0}, "We" ')
    lines.append(" ")
    formated_lines = format_freq_matrix(harm_freq, anharm_matrix)
    lines += formated_lines
    lines.append(" ")
    lines.append("0    'AMUA'")
    lines.append(" ")
    lines.append(f'{Egrain}   good  auto   450000. ')
    lines.append(f"'nochekstart'  {fname}.chk")
    lines.append(" ")

    # write file
    if verbose:
        print(f"{module.p:11} Writing to {datfile}")
    with open(datfile, "w") as f:
        f.writelines([line + "\n" for line in lines])

    return lines


def run_parsctst(
    datfile:Path = Path("parsctst.dat"),
    verbose:bool = False,
):
    cwd = datfile.parent.absolute()

    _default_datfile = "parsctst.dat"

    if datfile.name != _default_datfile:
        default_datfile = cwd / _default_datfile
        if verbose:
            print(f"{module.p:11} Copy {datfile} to {default_datfile}")
        default_datfile.write_text(datfile.read_text())

    command = f"cd {cwd}; " + config.machine.parsctst_command
    if verbose:
        print(f"{module.p:11} Run command: {command}")
    subprocess.call(command, shell=True)

def run_bdens(
    datfile:Path = Path("bdens.dat"),
    verbose:bool = False,
):
    cwd = datfile.parent.absolute()

    _default_datfile = "bdens.dat"

    if datfile.name != _default_datfile:
        default_datfile = cwd / _default_datfile
        if verbose:
            print(f"{module.p:11} Copy {datfile} to {default_datfile}")
        default_datfile.write_text(datfile.read_text())

    command = f"cd {cwd}; " + config.machine.bdens_command
    if verbose:
        print(f"{module.d:11} Run command: {command}")
    subprocess.call(command, shell=True)

def fix_crp_file(filename, add_text="    GOOD   VPT4A"):
    with open(filename) as f:
        lines = f.readlines()
    for n, line in enumerate(lines[1:]):
        if "INPUT DATA SUMMARY" in line:
            n_edit = n + 4
            # edit a line like this
            #      10.00   50000.00    8882.35    9053.73    2976.93
            break
    new_line = lines[n_edit][:-1] + add_text + os.linesep
    lines[n_edit] = new_line

    with open(filename, "w") as f:
        for line in lines:
            f.write(line)
        f.write(f"  {os.linesep}")
    return
