from pathlib import Path
import subprocess

from gausskit.multiwell.mominert import (
    run_mominert, write_mominert, get_rotor,
)
from gausskit.multiwell.densum import write_densum, run_densum
from gausskit.multiwell.thermo import write_thermo, run_thermo
from gausskit.multiwell.sctst import (
    write_parsctst, write_bdens, run_parsctst, run_bdens, fix_crp_file,
)

from gausskit.settings import Configuration

config = Configuration()


def run_PES_thermo(
    PES_data,
    thermo_methods,
    thermo_path=None,
    verbose=True,
    Egrain_line="10   3000   4000   50000",
):

    # Parse thermo_methods information
    thermo_hinderedrotor = thermo_methods["thermo_hinderedrotor"]
    thermo_anharm = thermo_methods["thermo_anharm"]

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
            for item in PES_data[PES_num]["PES_items"]:
                if PES_data[PES_num]["PES_items"][item]["ts"] == True:
                    sorted_freq = PES_data[PES_num]["PES_items"][item][
                        "Mol"
                    ].frequencies.copy()
                    sorted_freq.sort()
                    img_freq = sorted_freq[0]
                    if img_freq > 0:
                        print(f"Warning! Positive img_freq found {img_freq}")
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

    if thermo_path == None:
        thermo_path = "thermo"
    thermo_path = Path(thermo_path)
    if verbose:
        print(item_mol_name_list)

    # 1. prepare working directory
    if thermo_path.exists():
        pass
    else:
        thermo_path.mkdir()

    # 2 write and run mominert and densum
    for n, (dummy_name, Mol) in enumerate(zip(item_list, item_Mol_list)):
        mol = Mol

        # 2.1 write and run mominert
        datfile = f"{dummy_name}.coords"
        outfile = f"{dummy_name}.coords.out"
        write_mominert(mol, thermo_path / datfile, verbose=verbose)
        run_mominert(thermo_path / datfile, thermo_path / outfile, verbose=verbose)
        krot, ad_rot = get_rotor(thermo_path / outfile, verbose=verbose)

        # 2.2 write and run densum
        datfile = f"{dummy_name}.vib"
        outfile = f"{dummy_name}.dens"
        write_densum(
            mol,
            krot,
            ad_rot,
            fname=dummy_name,
            Egrain=Egrain_line,
            datfile=thermo_path / datfile, 
            verbose=verbose
        )
        run_densum(thermo_path / datfile, thermo_path / outfile, verbose=verbose)


    # 3 run bdens and/or parsctst if anharm
    # prepare bdens.dat or parsctst.dat
    # we only have one parsctst mission, so only one set of forw. backw. barrier
    if thermo_anharm:
        if verbose:
            print("---------------anharmonic(sctst)---------------")
        for n, (dummy_name, Mol) in enumerate(zip(item_list, item_Mol_list)):

            mol = Mol
            mol_name = item_mol_name_list[n]

            if len(mol.numbers) <= 1:
                print(f"Skip this one atom file {mol_name}")
                continue

            if mol.ts:
                # parsctst
                datfile = thermo_path / f"{dummy_name}.parsctst.dat"
                barrier = [forwards_barrier, backwards_barrier]
                write_parsctst(mol, dummy_name, barrier, Egrain_line, datfile, verbose)
                run_parsctst(datfile, verbose)
                fix_crp_file(thermo_path / f"{dummy_name}.crp")
                fix_crp_file(thermo_path / f"{dummy_name}.qcrp")

            else:
                # bdens
                datfile = thermo_path / f"{dummy_name}.bdens.dat"
                write_bdens(mol, dummy_name, Egrain_line, datfile, verbose)
                run_bdens(datfile, verbose)

    # 3.1 run mominert if we want internal hindered rotor
    # rewritre .coords file
    hindrot_item_reduced_mominert_dict = {}
    if thermo_hinderedrotor:
        for item in hindrot_item_Mol_dict:
            filename = _thermo_path / f"{item}.coords"
            with open(filename, "r") as f:
                lines = f.readlines()
            mol_hindrot = hindrot_item_Mol_dict[item].hinderedrotor

            # internal rotor coords information
            internal_rotor_coords = [""]
            for i_rotor in range(len(mol_hindrot._rotating_bonds)):
                internal_rotor_coords.append(
                    ", ".join(
                        [str(j + 1) for j in mol_hindrot._rotating_bonds[i_rotor]]
                    )
                )
                internal_rotor_coords.append(
                    str(len(mol_hindrot._rotating_groups[i_rotor]))
                )
                internal_rotor_coords.append(
                    ", ".join(
                        [str(j + 1) for j in mol_hindrot._rotating_groups[i_rotor]]
                    )
                )
                internal_rotor_coords.append("")
            internal_rotor_coords = [j + "\n" for j in internal_rotor_coords]

            # insert rotor coords information to the file
            n_line = lines.index(" 0 , 0\n")
            new_lines = lines[:n_line] + internal_rotor_coords + lines[n_line:]

            # rewrite coords file
            output = filename
            with open(output, "w") as f:
                f.writelines(new_lines)

            # run mominert
            command = (
                f"cd {_thermo_path}; "
                + config.machine.mominert_command
                + " {item}.coords"
            )
            subprocess.call(command, shell=True)

            # read moment of inertia from .co.out files
            filename = _thermo_path / "{item}.co.out"
            with open(filename, "r") as f:
                lines = f.readlines()
            mominert_lines = [
                line for line in lines if "REDUCED MOMENT OF INERTIA" in line
            ]
            reduced_moment_of_inertia = [
                float(mominert_line.split(":")[1].split()[0])
                for mominert_line in mominert_lines
            ]
            hindrot_item_reduced_mominert_dict[item] = reduced_moment_of_inertia

    # 4. prepare thermo input file reaction.dat
    datfile = "reaction.dat"
    write_thermo(
        PES_data,
        thermo_methods,
        thermo_path,
        hindrot_item_reduced_mominert_dict,
        datfile=thermo_path / datfile,
        verbose=verbose,
    )

    run_thermo(
        thermo_path,
        datfile=thermo_path / datfile,
        verbose=verbose,
    )

