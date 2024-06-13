import os
import shutil
import subprocess
from .thermo import fix_crp_file
from gausskit.gaussian.anharm import prepare_bdens, prepare_parsctst

def run_PES_multiwell(PES_data, multiwell_methods, multiwell_path=None, verbose=True, Egrain_line="10	3000	4000	50000"):
    
    # Parse multiwell_methods information
    multiwell_wells = multiwell_methods["multiwell_wells"]
    multiwell_products = multiwell_methods["multiwell_products"]
    multiwell_tss = multiwell_methods["multiwell_tss"]
    multiwell_anharm = multiwell_methods["multiwell_anharm"]

    # gather PES_info
    # item_list: Mol1, TS2, Mol3
    item_list =  []
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
            reverse_PES_num = list(PES_data.keys())[n+1]
            backwards_barrier = PES_data[reverse_PES_num]["reverse"]
            for item in PES_data[PES_num]["PES_items"]:
                if PES_data[PES_num]["PES_items"][item]["ts"] == True:
                    sorted_freq = PES_data[PES_num]["PES_items"][item]["Mol"].frequencies.copy()
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
    if os.path.exists(multiwell_path):
        pass
    else:
        os.mkdir(multiwell_path)

    # prepare DensData dir
    densdata_path = os.path.join(multiwell_path, "DensData")
    if os.path.exists(densdata_path):
        pass
    else:
        os.mkdir(densdata_path)
    
    for n, (dummy_name, Mol) in enumerate(zip(item_list,item_Mol_list)):
        mol = Mol
        dummy_logname = f"{dummy_name}.log"
        shutil.copy(mol.logpath, os.path.join(densdata_path,dummy_logname))

    # 2. write gauss2multi.cfg
    g2m_filepath = os.path.join(densdata_path, "gauss2multi.cfg")
    g2m_lines = ["KCAL", 
                "1", 
                "298",
                "ATM", 
                "1", 
                "1", 
                f"{Egrain_line}"]
    for n, (dummy_name, Mol) in enumerate(zip(item_list,item_Mol_list)):
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
    command = f"cd {densdata_path}; module load gcc/10.3.0 openmpi/4.1.1; echo N | gauss2multi"
    subprocess.run(command, shell=True, capture_output=True)

    # 3.5 run bdens and/or parsctst if anharm
    # prepare bdens.dat or parsctst.dat
    if multiwell_anharm:
        for n, (dummy_name, Mol) in enumerate(zip(item_list,item_Mol_list)):
            mol = Mol
            if mol.ts:
                # parsctst
                output_datname = f"{dummy_name}.parsctst.dat"
                print(output_datname)
                harm_freq = mol.frequencies
                anharm_matrix = mol.anharm_matrix
                barrier = [forwards_barrier, backwards_barrier]
                input_list = prepare_parsctst(dummy_name,harm_freq,anharm_matrix,Egrain_line,barrier)

                # write bdens.dat or parsctst.dat
                output = os.path.join(densdata_path, output_datname)
                with open(output, "w") as f:
                    f.writelines([line + "\n" for line in input_list])
                output = os.path.join(densdata_path, ".".join(output_datname.split(".")[1:]))
                with open(output, "w") as f:
                    f.writelines([line + "\n" for line in input_list])

                # run parsctst
                print("-----------------anharmonic-----------------")
                mol_name = item_mol_name_list[n]
                if mol.ts:
                    # run parsctst
                    command = f"cd {densdata_path}; module load gcc/10.3.0 openmpi/4.1.1; parsctst"
                    print(f"parsctst running for {mol_name}({dummy_name})")
                    subprocess.call(command, shell=True)
                    fix_crp_file(os.path.join(densdata_path, f"{dummy_name}.crp"))
                    fix_crp_file(os.path.join(densdata_path, f"{dummy_name}.qcrp"))
            else:
                # bdens
                pass

            
    # 4. prepare multiwell input file multiwell.dat

    reaction_lines = ["Gausskit generated. Be careful.", 
                    f"{Egrain_line}     1832960486", # a random seed 
                    "'ATM '  'KCAL'  'AMUA'",
                    " 298   298      !   <-  translational and initial vibrational temperatures.",
                    "1", # number of pressure
                    "1", # pressure
                    f"{len(multiwell_wells)}  {len(multiwell_products)}",
                    ]
    
    # formating well line
    n_channel = 1
    well_line = f"{n_channel}    'WELL{n_channel}'     0.00 {n_channel}" 

    # reaction_f = open(os.path.join(multiwell_path, "multiwell.dat"), "w")
    return
