from pathlib import Path
import os
import subprocess

import numpy as np

from gausskit.molecules import Molecules
from gausskit.multiwell.densum import get_degrees_of_freedom_lines
from gausskit.multiwell.mominert import get_rotor
from gausskit.settings import Configuration

config = Configuration()

module = "[Thermo]"

def write_thermo(
    PES_data,
    thermo_methods,
    thermo_path:Path,
    hindrot_item_reduced_mominert_dict,
    datfile:Path = Path("densum.dat"),
    verbose:bool = False,
):

    # Parse thermo_methods information
    thermo_tunneling = thermo_methods["thermo_tunneling"]
    thermo_hinderedrotor = thermo_methods["thermo_hinderedrotor"]
    thermo_anharm = thermo_methods["thermo_anharm"]
    thermo_adj_barrier = thermo_methods["thermo_adj_barrier"]
    thermo_temp = thermo_methods["thermo_temp"]
    thermo_pressure = thermo_methods["thermo_pressure"]

    if "default" in thermo_temp:
        temp = "200 300 400 500 600 800 1000 1200 1400 1600 1800 2000"
    else:
        temp = thermo_temp
    if "default" in thermo_pressure:
        pressure = "1"
    else:
        pressure = thermo_pressure

    # gather PES_info
    # item_list: Mol1, TS2, Mol3
    item_list = []
    # item_mol_name_list: HCFC133a, HCFC133a-OH_ts, Radical133a
    item_mol_name_list = []
    # item_Mol_list: 3 Molecules objects
    item_Mol_list = []
    for n, PES_num in enumerate(PES_data):
        for item in PES_data[PES_num]["PES_items"]:
            item_list.append(item)
            item_mol_name_list.append(PES_data[PES_num]["PES_items"][item]["mol_name"])
            Mol = PES_data[PES_num]["PES_items"][item]["Mol"]
            item_Mol_list.append(Mol)
        if PES_data[PES_num]["final_ts"] == True:
            forwards_barrier = PES_data[PES_num]["PES_energy"]
            reverse_PES_num = list(PES_data.keys())[n + 1]
            backwards_barrier = PES_data[reverse_PES_num]["reverse"]
            break

    # prepare hindered rot calculations
    # hindrot_item_Mol_dict: Molecule Objects if exists hindrot calculation
    if thermo_hinderedrotor:
        hindrot_item_Mol_dict = {}
        for n, PES_num in enumerate(PES_data):
            for item in PES_data[PES_num]["PES_items"]:
                if "Mol_hindrot" in PES_data[PES_num]["PES_items"][item]:
                    hindrot_item_Mol_dict[item] = PES_data[PES_num]["PES_items"][item][
                        "Mol_hindrot"
                    ]

    reaction_lines = [
        "KCAL   MCC",
        str(len(temp.split())),
        temp,
        f"{len(item_list)}",
    ]

    for n, (dummy_name, Mol) in enumerate(zip(item_list, item_Mol_list)):
        mol = Mol

        if mol.ts:
            # if no tunneling, set img_freq and backwards_barrier to 0
            if not thermo_tunneling:
                img_freq = 0
                backwards_barrier = 0
            reaction_lines.append(
                f"ctst    {dummy_name}    {forwards_barrier}   {-img_freq}   {backwards_barrier}"
            )
        else:
            reaction_lines.append(f"reac    {dummy_name}    0.0")

        reaction_lines.append(f"{mol.get_chemical_formula()}")
        reaction_lines.append(f"{mol.external_symmetry_number}   1   1")
        reaction_lines.append(f"{0.0:10} {mol.multiplicity}")

        mominert_outfile = f"{dummy_name}.coords.out"
        krot, ad_rot = get_rotor(thermo_path / mominert_outfile)

        # if run anharmonic thermo
        if thermo_anharm:
            # count the number of k-rotor and adiabatic rotor [NOT hindered rotor!!!]
            total_dof = 1 # read external file, this takes "one vibration"
            if np.abs(krot) > 1e-12:
                total_dof += 1
            if np.abs(ad_rot) > 1e-12:
                total_dof += 1
            
            # in case of one-atom species, don't write any degrees of freedom
            if len(mol.numbers) <= 1:
                total_dof = 0
                reaction_lines.append(f"{total_dof}   HAR   AMUA")
                reaction_lines.append(" ")
                continue

            # number of vibrations and rotations to be read in
            reaction_lines.append(f"{total_dof}   HAR   AMUA")

            # add read external file line
            if mol.ts:
                line = "1    crp   0.0      1.0     1     ! read external file"
            else:
                line = "1    qvb   0.0      1.0     1     ! read external file"
            reaction_lines.append(line)

            # add k-rotor and adiabatic rotor line
            # additional mode number begins with 2
            rottype = "qrot " if (krot < 11.0) else "rot  "
            n_dof = 1
            if np.abs(krot) > 1e-12:
                n_dof += 1
                reaction_lines.append(
                    f" {n_dof:3d}   {rottype:6} {krot:12.4f}   1.0   1   ! K-rotor"
                )

            rottype = "qrot " if (ad_rot < 11.0) else "rot  "
            if np.abs(ad_rot) > 1e-12:
                n_dof += 1
                reaction_lines.append(
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
            reaction_lines.append(f"{total_dof}   HAR   AMUA")
            dof_lines = get_degrees_of_freedom_lines(mol, krot, ad_rot)

            # if thermo_hinderedrotor and mol has hindered rotor
            # replace the selected vibration mode with the hindered rotor DOF
            if thermo_hinderedrotor and dummy_name in hindrot_item_Mol_dict:
                # vibration in xxx.therm file is One-based numbering
                mol = hindrot_item_Mol_dict[dummy_name]
                corrected_vibs = mol.hinderedrotor._corrected_vibs
                for n_index, n_vib in enumerate(corrected_vibs):
                    corr_rot = mol.hinderedrotor._reduced_moms[n_index]
                    rottype = "qrot " if (corr_rot < 11.0) else "rot  "
                    line = f"  # {n_vib:>3}{rottype:>6}{corr_rot:>9.4f}(from G16)"
                    line += f"{hindrot_item_reduced_mominert_dict[dummy_name][n_index]:>9.4f}(from Mominert)  "
                    line += f" {mol.hinderedrotor._symmetry_numbers[n_index]}   1"
                    dof_lines[n_vib] = line

            reaction_lines += dof_lines
        reaction_lines.append(f"  {os.linesep}")

    reaction_lines.append(f"  {os.linesep}")

    # prepare thermo.dat
    with open(datfile, "w") as f:
        for line in reaction_lines:
            f.write(f"{line} {os.linesep}")
        f.write(f"  {os.linesep}")

def run_thermo(
    datfile:Path = Path("thermo.dat"),
    verbose:bool = False,
):
    cwd = datfile.parent.absolute()

    command = f"cd {cwd}; " + config.machine.thermo_command + f" {datfile.name}"
    if verbose:
        print(f"{module:10} Run command: {command}")
    subprocess.call(command, shell=True)

