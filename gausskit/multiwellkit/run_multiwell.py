import os
import shutil
import subprocess

def run_PES_multiwell(PES_data, multiwell_methods, multiwell_path=None, verbose=True, Egrain_line="10	3000	4000	50000"):
    
    multiwell_wells = multiwell_methods["multiwell_wells"]
    multiwell_products = multiwell_methods["multiwell_products"]
    multiwell_tss = multiwell_methods["multiwell_tss"]
    multiwell_anharm = multiwell_methods["multiwell_anharm"]

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
    
    for n, (well_name, Mols) in enumerate(zip(item_list,item_Mols_list)):
        mol = Mols
        if mol.ts:
            dummy_name = f"TS{n+1}"
        else:
            dummy_name = f"WELL{n+1}"
        dummy_logname = f"{dummy_name}.log"
        shutil.copy(mol.logpath, os.path.join(densdata_path,dummy_logname))

    # 2. write gauss2multi.cfg
    g2m_filepath = os.path.join(densdata_path, "gauss2multi.cfg")
    g2m_lines = ["KCAL", 
                "12", 
                "200 300 400 500 600 800 1000 1200 1400 1600 1800 2000",
                "ATM", 
                "1", 
                "1", 
                f"{Egrain_line}"]
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
    command = f"cd {densdata_path}; module load gcc/10.3.0 openmpi/4.1.1; echo N | gauss2multi"
    subprocess.run(command, shell=True, capture_output=True)
    
    # 4. prepare multiwell input file multiwell.dat
    reaction_f = open(os.path.join(multiwell_path, "multiwell.dat"), "w")

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

    return