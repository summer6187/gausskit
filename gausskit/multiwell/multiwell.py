from pathlib import Path
import os
import shutil
import subprocess

from gausskit.multiwell.workflow import fix_crp_file
from gausskit.multiwell.sctst import write_bdens, write_parsctst
from gausskit.settings import Configuration
from gausskit._defaults import colliders, trail_line

config = Configuration()

def run_PES_multiwell(
    PES_data,
    multiwell_methods,
    multiwell_path=None,
    collider="N2",
    verbose=True,
    Egrain_line="10	3000	4000	50000",
):

    # Parse multiwell_methods information
    multiwell_wells = multiwell_methods["multiwell_wells"]
    multiwell_channels = multiwell_methods["multiwell_channels"]
    multiwell_anharm = multiwell_methods["multiwell_anharm"]

    # gather PES_info
    # item_list: Mol1, TS2, Mol3
    item_list = []
    # item_mol_name_list: HCFC133a, HCFC133a-OH_ts, Radical133a
    item_mol_name_list = []
    # item_Mol_list: 3 Molecules objects
    item_Mol_list = []
    # barrier list: include forward and backward barrier
    barrier_list = []
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
        else:
            # for well or product mol, set dummy barrier == 0
            forwards_barrier = 0
            backwards_barrier = 0
        barrier_list.append([forwards_barrier, backwards_barrier])

    # 1. prepare log files
    multiwell_path = Path(multiwell_path)
    if multiwell_path.exists():
        pass
    else:
        multiwell_path.mkdir(parents=True)

    # prepare DensData dir
    densdata_path = multiwell_path / "DensData"
    if densdata_path.exists():
        pass
    else:
        densdata_path.mkdir()

    for n, (dummy_name, Mol) in enumerate(zip(item_list, item_Mol_list)):
        mol = Mol
        dummy_logname = f"{dummy_name}.log"
        shutil.copy(mol.logpath, densdata_path / dummy_logname)

    # 2. write gauss2multi.cfg
    g2m_filepath = densdata_path / "gauss2multi.cfg"
    g2m_lines = ["KCAL", "1", "298", "ATM", "1", "1", f"{Egrain_line}"]
    for n, (dummy_name, Mol) in enumerate(zip(item_list, item_Mol_list)):
        mol = Mol
        if mol.ts:
            mol_line = f"{n+1}   {dummy_name}.log     TS"
        else:
            mol_line = f"{n+1}   {dummy_name}.log   WELL"
        g2m_lines.append(mol_line)

    with open(g2m_filepath, "w") as f:
        for line in g2m_lines:
            f.write(f"{line} {os.linesep}")
        f.write(f"  {os.linesep}")

    # 3. run gauss2multi
    # run gauss2multi needs an input N to choose not to overwrite gauss2multi.cfg file
    command = f"cd {densdata_path}; echo N | " + config.machine.gauss2multi_command
    subprocess.run(command, shell=True, capture_output=True)

    # # 3.5 run bdens and/or parsctst if anharm
    # # prepare bdens.dat or parsctst.dat
    # # we only have one parsctst mission, so only one set of forw. backw. barrier
    # if multiwell_anharm:
    #     for n, (dummy_name, Mol) in enumerate(zip(item_list, item_Mol_list)):
    #         mol = Mol
    #         if mol.ts:
    #             # parsctst
    #             output_datname = f"{dummy_name}.parsctst.dat"
    #             harm_freq = mol.frequencies
    #             anharm_matrix = mol.anharm_matrix
    #             barrier = [forwards_barrier, backwards_barrier]
    #             input_list = prepare_parsctst(
    #                 dummy_name, harm_freq, anharm_matrix, Egrain_line, barrier
    #             )
    #
    #         else:
    #             # bdens
    #             output_datname = f"{dummy_name}.bdens.dat"
    #             harm_freq = mol.frequencies
    #             anharm_matrix = mol.anharm_matrix
    #             input_list = prepare_bdens(
    #                 dummy_name, harm_freq, anharm_matrix, Egrain_line
    #             )
    #
    #         # write bdens.dat or parsctst.dat
    #         output = densdata_path / output_datname
    #         with open(output, "w") as f:
    #             f.writelines([line + "\n" for line in input_list])
    #         output = densdata_path / ".".join(output_datname.split(".")[1:])
    #         with open(output, "w") as f:
    #             f.writelines([line + "\n" for line in input_list])
    #
    #         # run bdens or parsctst
    #         print("-----------------anharmonic-----------------")
    #         mol_name = item_mol_name_list[n]
    #         if mol.ts:
    #             # run parsctst
    #             command = f"cd {densdata_path}; " + config.machine.parsctst_command
    #             print(f"parsctst running for {mol_name}({dummy_name})")
    #             subprocess.call(command, shell=True)
    #             fix_crp_file(densdata_path / f"{dummy_name}.crp")
    #             fix_crp_file(densdata_path / f"{dummy_name}.qcrp")
    #         else:
    #             # run bdens
    #             command = f"cd {densdata_path}; " + config.machine.bdens_command
    #             print(f"bdens running for {mol_name}({dummy_name})")
    #             subprocess.call(command, shell=True)

    # 4. prepare multiwell input file multiwell.dat

    reaction_lines = [
        "Gausskit generated. Be careful.",
        f"{Egrain_line}     1832960486",  # a random seed
        "\n",
        "'ATM'  'KCAL'  'AMUA'",
        "\n",
        " 298   298      !   <-  translational and initial vibrational temperatures.",
        "1",  # number of pressure
        "1",  # pressure
        f"{len(multiwell_wells)}  {len(multiwell_channels)}", # here we assume there is one and only one product for each channel
    ]
    
    n_mol = 0 # number of wells or products in this section
    # formatting well line
    for n_well in multiwell_wells:
        n_mol += 1
        mol_list = item_Mol_list[n_well]
        if type(mol_list)==list and len(mol_list) > 1:
            exit(f"This well is not unimolecule! Check it out! {mol_list}") # this is a shit code, rewrite it with expect
        mol = mol_list[0]
        dummy_name = "dummy_well" # we should get this from Mol
        relative_energy = 0.0
        rotational_parameter = 44.444
        external_symmetry_number = 1
        electronic_partition_function = 2
        chiral_stereoisomers = 1
        line = f"{n_mol}     {dummy_name}     {relative_energy}   {rotational_parameter} \
        {external_symmetry_number}   {electronic_partition_function}   {chiral_stereoisomers}"
        reaction_lines.append(line)
    
    # formatting product lines
    for (n_well, n_ts, n_product) in multiwell_channels:
        n_mol += 1
        dummy_name = "dummy_product" # we should get this from Mol
        relative_energy = 1.0
        line = f"{n_mol}    {dummy_name}    {relative_energy}"
        reaction_lines.append(line)
    
    # formatting collider model
    collider_line = colliders[collider]
    reaction_lines.append(collider_line)
    for n_well in multiwell_wells:
        lj_sigma = "5.17"
        lj_eps = "395.10"
        reaction_lines.append(f"{n_well}   {lj_sigma}   {lj_eps}   1        100.  0.0   0.0   0.0   0.0   0.0   0.0   0.0")
    reaction_lines.append(f"LJ")

    # formatting transition state lines
    reaction_lines.append("\n")
    reaction_lines.append(f"{len(multiwell_channels)}") # number of channels
    for (n_well, n_ts, n_product) in multiwell_channels:
        n_mol += 1
        dummy_name = "dummy_product" # we should get this from Mol
        rotational_parameter = 44.444
        reaction_critical_energy = 12.34
        line = f"{n_well}   {n_product}  {dummy_name}    {rotational_parameter}    1   2   1   0.00  "
        line += f"{reaction_critical_energy}  'rev' 'TUN' 'FAST' 'NOCENT' 'sum'"
        reaction_lines.append(line)

    # trial line
    reaction_lines.append("\n")
    reaction_lines.append(trail_line)

    multiwell_dat = multiwell_path / "multiwell.dat"
    print(f"Write to {multiwell_dat}")
    reaction_f = open(multiwell_dat, "w")
    reaction_f.write('\n'.join(reaction_lines) + '\n')
    reaction_f.close()


    # 5. run multiwell
    command = f"cd {multiwell_path}; " + config.machine.thermo_command + " multiwell.dat"
    subprocess.call(command, shell=True)
    return
