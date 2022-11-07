import subprocess
import pickle
import os
import shutil


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
                reaction_lines.append(f"ctst    {dummy_name}    {forwards_barrier}   {img_freq}   {backwards_barrier}")
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