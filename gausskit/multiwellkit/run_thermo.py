import subprocess
import pickle
import os
import shutil
import numpy as np


def run_thermo(dataset, mol_list, thermo_path, **calc_para):

    method = calc_para["method"]
    forwards_barrier = calc_para["forwards_barrier"]
    img_freq = calc_para["img_freq"]
    backwards_barrier = calc_para["backwards_barrier"]


    # 1. prepare log files
    if os.path.exists(thermo_path):
        pass
    else:
        os.mkdir(thermo_path)

    for n,well_name in enumerate(mol_list):
        mol = dataset[well_name][method]
        if mol.ts:
            dummy_name = f"TS{n+1}"
        else:
            dummy_name = f"WELL{n+1}"
        dummy_logname = f"{dummy_name}.log"
        shutil.copy(mol.logpath, os.path.join(thermo_path,dummy_logname))


    # 2. write gauss2multi.cfg
    g2m_filepath = os.path.join(thermo_path, "gauss2multi.cfg")
    g2m_lines = ["KCAL", 
                "12", 
                "200 300 400 500 600 800 1000 1200 1400 1600 1800 2000",
                "ATM", 
                "1", 
                "1", 
                "10.    3000     4000     200000."]
    for n,well_name in enumerate(mol_list):
        mol = dataset[well_name][method]
        if mol.ts:
            mol_line = f"{n+1}   TS{n+1}.log     TS"
        else:
            mol_line = f"{n+1}   WELL{n+1}.log   WELL"
        g2m_lines.append(mol_line)

    with open(g2m_filepath, "w") as f:
        for line in g2m_lines:
            f.write(f"{line} {os.linesep}")
        f.write(f"  {os.linesep}")
        

    # 3. run gauss2multi
    command = f"cd {thermo_path}; module load gcc/10.3.0 openmpi/4.1.1; echo N | gauss2multi"
    subprocess.call(command, shell=True)


    # 4. prepare thermo input file reaction.dat
    reaction_f = open(os.path.join(thermo_path, "reaction.dat"), "w")

    reaction_lines = ["KCAL   MCC", 
                    "12", 
                    "200 300 400 500 600 800 1000 1200 1400 1600 1800 2000",
                    f"{len(mol_list)}",
                    ]

    for n,well_name in enumerate(mol_list):
        mol = dataset[well_name][method]
        if mol.ts:
            dummy_name = f"TS{n+1}"
        else:
            dummy_name = f"WELL{n+1}"
        dummy_thermname = f"{dummy_name}.therm"

        with open(os.path.join(thermo_path, dummy_thermname)) as f:
            mol_lines = f.readlines()
            if mol.ts:
                print()
                reaction_lines.append(f"ctst    {dummy_name}    {forwards_barrier}   {-img_freq}   {backwards_barrier}")
            else:
                reaction_lines.append(f"reac    {dummy_name}    0.0")
            for line in mol_lines[5:]:
                if line[-1:] == "\n":
                    reaction_lines.append(line[:-1])
                else:
                    reaction_lines.append(line)

    for line in reaction_lines:
        reaction_f.write(f"{line} {os.linesep}")
    reaction_f.write(f"  {os.linesep}")
    reaction_f.close()

    # 5. run thermo
    command = f"cd {thermo_path}; module load gcc/10.3.0 openmpi/4.1.1; thermo reaction.dat"
    subprocess.call(command, shell=True)


def run_PES_thermo(PES_data, thermo_methods, thermo_path=None, verbose=True):

    thermo_tunneling = thermo_methods["thermo_tunneling"]
    thermo_hinderedrotor = thermo_methods["thermo_hinderedrotor"]

    # gather PES_info
    item_list =  []
    item_Mols_list = []
    for n, PES_num in enumerate(PES_data):
        for item in PES_data[PES_num]["PES_items"]:
            item_list.append(item)
            Mols = PES_data[PES_num]["PES_items"][item]["Mols"]
            item_Mols_list.append(Mols)
        if PES_data[PES_num]["final_ts"] == True:
            forwards_barrier = PES_data[PES_num]["PES_energy"]
            reverse_PES_num = list(PES_data.keys())[n+1]
            backwards_barrier = PES_data[reverse_PES_num]["reverse"]
            for item in PES_data[PES_num]["PES_items"]:
                if PES_data[PES_num]["PES_items"][item]["ts"] == True:
                    sorted_freq = PES_data[PES_num]["PES_items"][item]["Mols"].frequencies.copy()
                    sorted_freq.sort()
                    img_freq = sorted_freq[0]
                    if img_freq > 0:
                        print(f"Warning! Positive img_freq found {img_freq}")
            break
    
    # prepare hindered rot calculations
    if thermo_hinderedrotor:
        hindrot_item_Mols_dict = {}
        for n, PES_num in enumerate(PES_data):
            for item in PES_data[PES_num]["PES_items"]:
                if "Mols_hindrot" in PES_data[PES_num]["PES_items"][item]:
                    hindrot_item_Mols_dict[item] = PES_data[PES_num]["PES_items"][item]["Mols_hindrot"]

    if thermo_path == None:
        thermo_path = "testcases/thermo_test"
    if verbose:
        print(item_list)

    # 1. prepare log files
    if os.path.exists(thermo_path):
        pass
    else:
        os.mkdir(thermo_path)

    for n, (well_name, Mols) in enumerate(zip(item_list,item_Mols_list)):
        mol = Mols
        if mol.ts:
            dummy_name = f"TS{n+1}"
        else:
            dummy_name = f"WELL{n+1}"
        dummy_logname = f"{dummy_name}.log"
        shutil.copy(mol.logpath, os.path.join(thermo_path,dummy_logname))


    # 2. write gauss2multi.cfg
    g2m_filepath = os.path.join(thermo_path, "gauss2multi.cfg")
    g2m_lines = ["KCAL", 
                "12", 
                "200 300 400 500 600 800 1000 1200 1400 1600 1800 2000",
                "ATM", 
                "1", 
                "1", 
                "10.    3000     4000     200000."]
    for n, (well_name, Mols) in enumerate(zip(item_list,item_Mols_list)):
        mol = Mols
        if mol.ts:
            mol_line = f"{n+1}   TS{n+1}.log     TS"
        else:
            mol_line = f"{n+1}   WELL{n+1}.log   WELL"
        g2m_lines.append(mol_line)

    with open(g2m_filepath, "w") as f:
        for line in g2m_lines:
            f.write(f"{line} {os.linesep}")
        f.write(f"  {os.linesep}")
        

    # 3. run gauss2multi
    command = f"cd {thermo_path}; module load gcc/10.3.0 openmpi/4.1.1; echo N | gauss2multi"
    subprocess.run(command, shell=True, capture_output=True)


    # 4. prepare thermo input file reaction.dat
    reaction_f = open(os.path.join(thermo_path, "reaction.dat"), "w")

    reaction_lines = ["KCAL   MCC", 
                    "12", 
                    "200 300 400 500 600 800 1000 1200 1400 1600 1800 2000",
                    f"{len(item_list)}",
                    ]

    for n, (well_name, Mols) in enumerate(zip(item_list,item_Mols_list)):
        mol = Mols
        if mol.ts:
            dummy_name = f"TS{n+1}"
        else:
            dummy_name = f"WELL{n+1}"
        dummy_thermname = f"{dummy_name}.therm"

        with open(os.path.join(thermo_path, dummy_thermname)) as f:
            mol_lines = f.readlines()
            if mol.ts:
                # if no tunneling, set img_freq and backwards_barrier to 0
                if not thermo_tunneling:
                    img_freq = 0
                    backwards_barrier = 0
                reaction_lines.append(f"ctst    {dummy_name}    {forwards_barrier}   {-img_freq}   {backwards_barrier}")
            else:
                reaction_lines.append(f"reac    {dummy_name}    0.0")
            for line in mol_lines[5:]:
                # if thermo_hinderedrotor and mol has hindered rotor
                if thermo_hinderedrotor and well_name in hindrot_item_Mols_dict:
                    # vibration in xxx.therm file is One-based numbering
                    mol_hindrot = hindrot_item_Mols_dict[well_name]
                    corrected_vibs = mol_hindrot.hinderedrotor._corrected_vibs
                    corrected_vibs = [n+1 for n in corrected_vibs]
                    # match lines like " 1   vib        56.7966  0.0    1"
                    if len(line.split()) > 1:
                        if line.split()[0].isdigit() and int(line.split()[0]) in corrected_vibs:
                            #  and line.split()[1] == "vib":
                            vib_index = int(line.split()[0])
                            itemindex = corrected_vibs.index(vib_index)
                            vib_type = "qrot"
                            line = f"{vib_index:>3}{vib_type:>6}{mol_hindrot.hinderedrotor._reduced_moms[itemindex]:>15}  1.0    1"
                if line[-1:] == "\n":
                    reaction_lines.append(line[:-1])
                else:
                    reaction_lines.append(line)

        reaction_lines.append(f"  {os.linesep}")

    for line in reaction_lines:
        reaction_f.write(f"{line} {os.linesep}")
    reaction_f.write(f"  {os.linesep}")
    reaction_f.close()

    # 5. run thermo
    command = f"cd {thermo_path}; module load gcc/10.3.0 openmpi/4.1.1; thermo reaction.dat"
    subprocess.call(command, shell=True)

if __name__ == "__main__":
    # this is a test case
    with open("testcases/database.pickle", "rb") as f:
        dataset = pickle.load(f)

    mol_list = ["HCFC133a", "OH", "HCFC133a-OH_ts"]
    thermo_path = "testcases/thermo_test"
    calc_para = {
        "method": "G4",
        "forwards_barrier": 1.111,
        "backwards_barrier": 0.0,
        "img_freq": 0.0
    }
    
    run_thermo(dataset, mol_list, thermo_path, **calc_para)