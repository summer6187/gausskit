from pathlib import Path
import os
import subprocess
from typing import Optional

import numpy as np

from gausskit.molecules import Molecules
from gausskit.multiwell.densum import get_degrees_of_freedom_lines
from gausskit.multiwell.mominert import get_rotor
from gausskit.settings import Configuration

config = Configuration()

module = "[Thermo]"

def get_thermo_head_lines(temp, n_species):
    """Return header lines for a ``thermo`` input file.

    Args:
        temp (str): Temperature grid specification.
        n_species (int): Number of species blocks to follow.

    Returns:
        list[str]: The header lines.
    """

    head_lines = [
        "KCAL   MCC",
        str(len(temp.split())),
        temp,
        f"{n_species}",
    ]

    return head_lines


def get_thermo_lines(
    mol:Molecules,
    dummy_name:str,
    thermo_path:Path,
    mol_type:Optional[str] = None,
    forwards_barrier:float = 0.0,
    backwards_barrier:float = 0.0,
    if_ktools:float = False,
    if_tunneling:bool = False,
    if_anharm:bool = False,
    if_hinderedrotor:bool = False,
    hindrot_item_Mol_dict:dict = {},
    hindrot_item_reduced_mominert_dict:dict = {},
):
    """Generate the block describing ``mol`` for a ``thermo`` input file.

    Args:
        mol (Molecules): Molecule being described.
        dummy_name (str): Identifier used in the input file.
        thermo_path (Path): Directory for auxiliary files.
        mol_type (str, optional): ``'reac'``, ``'prod'`` or ``'ctst'``.
        forwards_barrier (float, optional): Forward barrier height in kcal/mol.
        backwards_barrier (float, optional): Reverse barrier height.
        if_ktools (bool, optional): Use ktools formatting. Defaults to ``False``.
        if_tunneling (bool, optional): Include tunnelling information.
        if_anharm (bool, optional): Treat anharmonic corrections.
        if_hinderedrotor (bool, optional): Include hindered rotors.
        hindrot_item_Mol_dict (dict, optional): Mapping of dummy names to
            hindered rotor molecules.

    Returns:
        list[str]: Lines describing ``mol``.
    """
    lines = []

    if mol_type is not None:
        # ktools path passes forwards_barrier as a preformatted "energy   bond" string;
        # the thermo path passes a float. Handle both.
        bar = forwards_barrier if isinstance(forwards_barrier, str) else f"{forwards_barrier:.4f}"
        lines.append(f"{mol_type}    {dummy_name}    {bar}")
    else:
        if mol.ts:
            # if no tunneling, set img_freq and backwards_barrier to 0
            if not if_tunneling:
                img_freq = 0
                backwards_barrier = 0
            else:
                freq = mol.frequencies
                img_freq_list = freq[freq<0]
                assert len(img_freq_list) == 1, f"Something wrong with img frequency {img_freq_list}"
                img_freq = img_freq_list[0]
            lines.append(
                f"ctst    {dummy_name}    {forwards_barrier:.4f}   {-img_freq:.4f}   {backwards_barrier:.4f}"
            )
        else:
            lines.append(f"reac    {dummy_name}    {forwards_barrier:.4f}")

    lines.append(f"{mol.get_chemical_formula()}")
    if if_ktools:
        lines.append("1. Comment line")
        lines.append("2. Comment line")
        lines.append("3. Comment line")
    lines.append(f"{mol.external_symmetry_number}   {mol.optical_isomers}   1")
    lines.append(f" {0.0:<10} {mol.multiplicity}")

    mominert_outfile = f"{dummy_name}.coords.out"
    krot, ad_rot = get_rotor(thermo_path / mominert_outfile)

    # if run anharmonic thermo
    if if_anharm:
        # count the number of k-rotor and adiabatic rotor [NOT hindered rotor!!!]
        total_dof = 1 # read external file, this takes "one vibration"
        if np.abs(krot) > 1e-12:
            total_dof += 1
        if np.abs(ad_rot) > 1e-12:
            total_dof += 1

        # in case of one-atom species, don't write any degrees of freedom
        if len(mol.numbers) <= 1:
            total_dof = 0
            lines.append(f"{total_dof}   HAR   AMUA")
            lines.append(" ")
            return lines

        # number of vibrations and rotations to be read in
        lines.append(f"{total_dof}   HAR   AMUA")

        # add read external file line
        if mol.ts:
            line = "1    crp   0.0      1.0     1     ! read external file"
        else:
            line = "1    qvb   0.0      1.0     1     ! read external file"
        lines.append(line)

        # add k-rotor and adiabatic rotor line
        # additional mode number begins with 2
        if if_ktools:
            rottype = "kro"
        else:
            rottype = "qrot " if (krot < 11.0) else "rot  "
        n_dof = 1
        if np.abs(krot) > 1e-12:
            n_dof += 1
            lines.append(
                f" {n_dof:3d}   {rottype:6} {krot:12.4f}   1.0   1   ! K-rotor"
            )

        if if_ktools:
            rottype = "jro"
        else:
            rottype = "qrot " if (ad_rot < 11.0) else "rot  "
        if np.abs(ad_rot) > 1e-12:
            n_dof += 1
            lines.append(
                f" {n_dof:3d}   {rottype:6} {ad_rot:12.4f}   1.0   2   ! 2D adiabatic rotor"
            )

    # with normal modes, lines will be the same as fname.vibs
    else:
        # get all non imaginary frequencies
        freq = mol.frequencies
        nonimg_freq = freq[freq>0]
        total_dof = len(nonimg_freq)
        if np.abs(krot) > 1e-12:
            total_dof += 1
        if np.abs(ad_rot) > 1e-12:
            total_dof += 1
        lines.append(f"{total_dof}   HAR   AMUA")
        dof_lines = get_degrees_of_freedom_lines(mol, krot, ad_rot, if_ktools=if_ktools)

        # if thermo_hinderedrotor and mol has hindered rotor
        # replace the selected vibration mode with the hindered rotor DOF
        if if_hinderedrotor and dummy_name in hindrot_item_Mol_dict:
            # vibration in xxx.therm file is One-based numbering
            mol = hindrot_item_Mol_dict[dummy_name]
            corrected_vibs = mol.hinderedrotor._corrected_vibs
            for n_index, n_vib in enumerate(corrected_vibs):
                corr_rot = mol.hinderedrotor._reduced_moms[n_index]
                rottype = "qrot " if (corr_rot < 11.0) else "rot  "
                line = f"  # {n_vib:>3}{rottype:>6}{corr_rot:>9.4f}(from G16)"
                line += f"{hindrot_item_reduced_mominert_dict[dummy_name][n_index]:>9.4f}(from Mominert)  " # FIXME: wtf???
                line += f" {mol.hinderedrotor._symmetry_numbers[n_index]}   1"
                dof_lines[n_vib] = line

        lines += dof_lines
    lines.append(f"  {os.linesep}")
    return lines

def write_thermo(
    PES_data,
    thermo_methods,
    thermo_path:Path,
    hindrot_item_reduced_mominert_dict,
    datfile:Path = Path("densum.dat"),
    verbose:bool = False,
):
    """Write the main ``thermo`` input file describing the PES.

    Args:
        PES_data (dict): Potential energy surface data.
        thermo_methods (dict): Dictionary of thermo options.
        thermo_path (Path): Directory where files are written.
        hindrot_item_reduced_mominert_dict (dict): Reduced moments for hindered
            rotors.
        datfile (Path, optional): Output file name. Defaults to ``densum.dat``.
        verbose (bool, optional): Print progress messages. Defaults to ``False``.

    Returns:
        None
    """

    # Parse thermo_methods information
    if_tunneling = thermo_methods["tunneling"]
    if_hinderedrotor = thermo_methods["hinderedrotor"]
    if_anharm = thermo_methods["anharm"]
    thermo_methods["adj_barrier"]
    thermo_temp = thermo_methods["temperatures"]
    thermo_pressure = thermo_methods["pressures"]

    if "default" in thermo_temp:
        temp = "200 300 400 500 600 800 1000 1200 1400 1600 1800 2000"
    else:
        temp = thermo_temp
    if "default" in thermo_pressure:
        pass
    else:
        pass

    # gather PES_info
    # item_list: Mol1, TS2, Mol3
    item_list = []
    # item_mol_name_list: HCFC133a, HCFC133a-OH_ts, Radical133a
    item_mol_name_list = []
    # item_Mol_list: 3 Molecules objects
    item_Mol_list = []
    # forward barrier list: 0.0, 1.78
    forwards_barrier_list = []
    for n, PES_num in enumerate(PES_data):
        for item in PES_data[PES_num]["PES_items"]:
            item_list.append(item)
            item_mol_name_list.append(PES_data[PES_num]["PES_items"][item]["mol_name"])
            Mol = PES_data[PES_num]["PES_items"][item]["Mol"]
            item_Mol_list.append(Mol)
            forwards_barrier = PES_data[PES_num]["PES_energy"]
            forwards_barrier_list.append(forwards_barrier)
        if PES_data[PES_num]["final_ts"]:
            reverse_PES_num = list(PES_data.keys())[n + 1]
            backwards_barrier = PES_data[reverse_PES_num]["reverse"]
            break

    # prepare hindered rot calculations
    # hindrot_item_Mol_dict: Molecule Objects if exists hindrot calculation
    hindrot_item_Mol_dict = {}
    if if_hinderedrotor:
        for n, PES_num in enumerate(PES_data):
            for item in PES_data[PES_num]["PES_items"]:
                if "Mol_hindrot" in PES_data[PES_num]["PES_items"][item]:
                    hindrot_item_Mol_dict[item] = PES_data[PES_num]["PES_items"][item][
                        "Mol_hindrot"
                    ]

    reaction_lines = get_thermo_head_lines(temp, len(item_list))

    for n, (dummy_name, Mol, forwards_barrier) in enumerate(zip(item_list, item_Mol_list, forwards_barrier_list)):
        mol = Mol

        lines = get_thermo_lines(
            mol,
            dummy_name = dummy_name,
            thermo_path = thermo_path,
            forwards_barrier = forwards_barrier,
            backwards_barrier = backwards_barrier,
            if_tunneling = if_tunneling,
            if_anharm = if_anharm,
            if_hinderedrotor = if_hinderedrotor,
            hindrot_item_Mol_dict = hindrot_item_Mol_dict,
            hindrot_item_reduced_mominert_dict = hindrot_item_reduced_mominert_dict,
        )

        reaction_lines += lines

    reaction_lines.append(f"  {os.linesep}")

    # prepare thermo.dat
    with open(datfile, "w") as f:
        for line in reaction_lines:
            f.write(f"{line} {os.linesep}")
        f.write(f"  {os.linesep}")

def write_single_thermo(
    temp,
    mol:Molecules,
    dummy_name:str,
    thermo_path:Path = Path("thermo"),
    verbose:bool = False,
):
    """Write a stand-alone ``thermo`` input file for ``mol``.

    Args:
        temp (str): Temperature grid string.
        mol (Molecules): Molecule to be processed.
        dummy_name (str): Identifier prefix.
        thermo_path (Path, optional): Output directory. Defaults to ``thermo``.
        verbose (bool, optional): Emit progress information. Defaults to
            ``False``.

    Returns:
        None
    """

    reaction_lines = get_thermo_head_lines(temp, 1)

    lines = get_thermo_lines(
        mol,
        dummy_name = dummy_name,
        thermo_path = thermo_path,
        mol_type = "none"
    )

    reaction_lines += lines

    reaction_lines.append(f"  {os.linesep}")

    # prepare thermo.dat
    data_name = f"{dummy_name}.therm"
    datfile = thermo_path.absolute() / data_name
    if verbose:
        print(f"{module:11} Write to: {datfile}")
    with open(datfile, "w") as f:
        for line in reaction_lines:
            f.write(f"{line} {os.linesep}")
        f.write(f"  {os.linesep}")
    return


def run_thermo(
    datfile:Path = Path("thermo.dat"),
    outfile:Path = None,
    verbose:bool = False,
):
    """Execute the ``thermo`` program using ``datfile`` as input.

    Args:
        datfile (Path): ``thermo`` input file.
        outfile (Path, optional): Where to move the output. Defaults to the
            same name with ``.out`` extension.
        verbose (bool, optional): Display executed command. Defaults to
            ``False``.

    Returns:
        None
    """

    cwd = datfile.parent.absolute()

    command = f"cd {cwd}; " + config.machine.thermo_command + f" {datfile.name}"
    if verbose:
        print(f"{module:11} Run command: {command}")
    subprocess.call(command, shell=True)

    if outfile is not None:
        # get default output file name
        _datname = str(datfile.absolute())
        default_outfile = Path(_datname[:len(_datname)-4] + ".out")

        # make sure the output file exists
        assert default_outfile.exists(), f"{default_outfile} doesn't exists!"

        # move the default output file to targeted outfile
        if verbose:
            print(f"{module:11} Write to {outfile}")
        default_outfile.rename(outfile.absolute())

def read_electronic_partition_function(
    outfile:Path = Path("thermo.out"),
    verbose:bool = False,
) -> float:
    """Read the temperature independent electronic partition function.

    Args:
        outfile (Path): Output file from ``thermo``.
        verbose (bool, optional): Print diagnostic information. Defaults to
            ``False``.

    Returns:
        float: The electronic partition function.
    """
    with open(outfile, "r") as f:
        lines = f.readlines()

    temp_list = []
    qele_list = []

    start_line_number = 0
    for n_line, line in enumerate(lines):
        if "Qelectr" in line:
            start_line_number = n_line + 1
            break

    for line in lines[start_line_number:]:
        data = line.split()
        if data != []:
            temp_list.append(float(data[0]))
            qele_list.append(float(data[8]))
        else:
            break

    # Qelectr is temperature dependent, but usually the temperature 
    # dependence is neglagible, and the all the numbers are same
    qele = np.array(qele_list)
    assert np.allclose(qele, np.ones_like(qele) * qele[0]), f"Electronic partition function is T dependent! {qele}"

    return float(qele[0])

