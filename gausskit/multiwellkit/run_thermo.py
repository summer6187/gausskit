import subprocess
import os
import shutil

from gausskit.get_anharm_input import prepare_bdens, prepare_parsctst
from gausskit.settings import Configuration

config = Configuration()

def fix_crp_file(filename, add_text="    GOOD   VPT4A"):
    with open(filename) as f:
        lines = f.readlines()
    for n,line in enumerate(lines[1:]):
        if "INPUT DATA SUMMARY" in line:
            n_edit = n+4
            # edit a line like this
            #      10.00   50000.00    8882.35    9053.73    2976.93   
            break
    new_line = lines[n_edit][:-1] + add_text + os.linesep
    lines[n_edit] = new_line

    with open(filename ,"w") as f:
        for line in lines:
            f.write(line)
        f.write(f"  {os.linesep}")
    return


def run_PES_thermo(PES_data, thermo_methods, thermo_path=None, verbose=True, Egrain_line="10	3000	4000	50000"):

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
    item_list =  []
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
            reverse_PES_num = list(PES_data.keys())[n+1]
            backwards_barrier = PES_data[reverse_PES_num]["reverse"]
            for item in PES_data[PES_num]["PES_items"]:
                if PES_data[PES_num]["PES_items"][item]["ts"] == True:
                    sorted_freq = PES_data[PES_num]["PES_items"][item]["Mol"].frequencies.copy()
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
                    hindrot_item_Mol_dict[item] = PES_data[PES_num]["PES_items"][item]["Mol_hindrot"]

    if thermo_path == None:
        thermo_path = "testcases/thermo_test"
    if verbose:
        print(item_mol_name_list)

    # 1. prepare log files
    if os.path.exists(thermo_path):
        pass
    else:
        os.mkdir(thermo_path)

    for n, (dummy_name, Mol) in enumerate(zip(item_list,item_Mol_list)):
        mol = Mol
        dummy_logname = f"{dummy_name}.log"
        shutil.copy(mol.logpath, os.path.join(thermo_path,dummy_logname))


    # 2. write gauss2multi.cfg
    g2m_filepath = os.path.join(thermo_path, "gauss2multi.cfg")

    g2m_lines = ["KCAL", 
                str(len(temp.split())), 
                temp,
                "ATM", 
                str(len(pressure.split())), 
                pressure, 
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
    command = f"cd {thermo_path}; echo N | " + config.machine.gauss2multi_command
    subprocess.run(command, shell=True, capture_output=True)

    # 3.5 run bdens and/or parsctst if anharm
    # prepare bdens.dat or parsctst.dat
    # we only have one parsctst mission, so only one set of forw. backw. barrier
    if thermo_anharm:
        for n, (dummy_name, Mol) in enumerate(zip(item_list,item_Mol_list)):
            mol = Mol
            if mol.ts:
                # parsctst
                output_datname = f"{dummy_name}.parsctst.dat"
                harm_freq = mol.frequencies
                anharm_matrix = mol.anharm_matrix
                barrier = [forwards_barrier, backwards_barrier]
                input_list = prepare_parsctst(dummy_name,harm_freq,anharm_matrix,Egrain_line,barrier)

            else:
                # bdens
                output_datname = f"{dummy_name}.bdens.dat"
                harm_freq = mol.frequencies
                anharm_matrix = mol.anharm_matrix
                input_list = prepare_bdens(dummy_name,harm_freq,anharm_matrix,Egrain_line)
            
            # write bdens.dat or parsctst.dat
            output = os.path.join(thermo_path, output_datname)
            with open(output, "w") as f:
                f.writelines([line + "\n" for line in input_list])
            output = os.path.join(thermo_path, ".".join(output_datname.split(".")[1:]))
            with open(output, "w") as f:
                f.writelines([line + "\n" for line in input_list])
            
            # run bdens or parsctst
            print("-----------------anharmonic-----------------")
            mol_name = item_mol_name_list[n]
            if mol.ts:
                # run parsctst
                command = f"cd {thermo_path}; " + config.machine.parsctst_command
                print(f"parsctst running for {mol_name}({dummy_name})")
                subprocess.call(command, shell=True)
                fix_crp_file(os.path.join(thermo_path, f"{dummy_name}.crp"))
                fix_crp_file(os.path.join(thermo_path, f"{dummy_name}.qcrp"))
            else:
                # run bdens
                command = f"cd {thermo_path}; " + config.machine.bdens_command
                print(f"bdens running for {mol_name}({dummy_name})")
                subprocess.call(command, shell=True)
            
    # 3.5.5 moment of inertia
    # rewritre .coords file
    if thermo_hinderedrotor:
        hindrot_item_reduced_mominert_dict = {}
        for item in hindrot_item_Mol_dict:
            filename = os.path.join(thermo_path, item + ".coords")
            with open(filename, "r") as f:
                lines = f.readlines()
            mol_hindrot = hindrot_item_Mol_dict[item].hinderedrotor

            # internal rotor coords information
            internal_rotor_coords = ['']
            for i_rotor in range(len(mol_hindrot._rotating_bonds)):
                internal_rotor_coords.append(', '.join([str(j+1) for j in mol_hindrot._rotating_bonds[i_rotor]]))
                internal_rotor_coords.append(str(len(mol_hindrot._rotating_groups[i_rotor])))
                internal_rotor_coords.append(', '.join([str(j+1) for j in mol_hindrot._rotating_groups[i_rotor]]))
                internal_rotor_coords.append('')
            internal_rotor_coords = [j + '\n' for j in internal_rotor_coords]

            # insert rotor coords information to the file
            n_line = lines.index(' 0 , 0\n')
            new_lines = lines[:n_line] + internal_rotor_coords + lines[n_line:]

            # rewrite coords file
            output = filename
            with open(output, "w") as f:
                f.writelines(new_lines)
    
            # run mominert
            command = f"cd {thermo_path}; " + config.machine.mominert_command + " {item}.coords"
            subprocess.call(command, shell=True)

            # read moment of inertia from .co.out files
            filename = os.path.join(thermo_path, item + ".co.out")
            with open(filename, "r") as f:
                lines = f.readlines()
            mominert_lines = [line for line in lines if "REDUCED MOMENT OF INERTIA" in line]
            reduced_moment_of_inertia = [float(mominert_line.split(":")[1].split()[0]) for mominert_line in mominert_lines]
            hindrot_item_reduced_mominert_dict[item] = reduced_moment_of_inertia

    # 4. prepare thermo input file reaction.dat
    reaction_f = open(os.path.join(thermo_path, "reaction.dat"), "w")

    reaction_lines = ["KCAL   MCC", 
                    str(len(temp.split())), 
                    temp,
                    f"{len(item_list)}",
                    ]

    # shit mountain!!
    for n, (dummy_name, Mol) in enumerate(zip(item_list,item_Mol_list)):
        mol = Mol
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
            # if run anharmonic thermo
            if thermo_anharm:
                # count the number of k-rotor and adiabatic rotor [NOT hindered rotor!!!]
                n_rot = 0
                for line in mol_lines[5:]:
                    if "rot" in line:
                        n_rot += 1
                
                # append normal lines before HAR in line
                for line in mol_lines[5:]:
                    if "HAR" in line:
                        har_line_list = line.split()
                        # set degrees of freedom to 1 external file + number of k-rotor and adiabatic rotor
                        har_line_list[0] = str(n_rot + 1)
                        har_line = "    ".join(har_line_list)
                        reaction_lines.append(har_line)
                        break
                
                    if line[-1:] == "\n":
                        reaction_lines.append(line[:-1])
                    else:
                        reaction_lines.append(line)
                
                # add read external file line 
                if mol.ts:
                    line = "1    crp   0.0      1.0     1     ! read external file"
                else:
                    line = "1    qvb   0.0      1.0     1     ! read external file"
                reaction_lines.append(line)

                # add k-rotor and adiabatic rotor line
                # additional mode number begins with 2
                n_mode = 2
                for line in mol_lines[5:]:
                    if "rot" in line:
                        rot_line_list = line.split()
                        rot_line_list[0] = str(n_mode)
                        rot_line = "    ".join(rot_line_list)
                        reaction_lines.append(rot_line)
                        n_mode += 1
            else:
                for line in mol_lines[5:]:    
                    # if thermo_hinderedrotor and mol has hindered rotor
                    # if not, line will not be modified
                    if thermo_hinderedrotor and dummy_name in hindrot_item_Mol_dict:
                        # vibration in xxx.therm file is One-based numbering
                        mol = hindrot_item_Mol_dict[dummy_name]
                        corrected_vibs = mol.hinderedrotor._corrected_vibs
                        corrected_vibs = [n+1 for n in corrected_vibs]
                        # match lines like " 1   vib        56.7966  0.0    1"
                        if len(line.split()) > 1:
                            if line.split()[0].isdigit() and line.split()[1] == "vib" and int(line.split()[0]) in corrected_vibs:
                                #  and line.split()[1] == "vib":
                                vib_index = int(line.split()[0])
                                itemindex = corrected_vibs.index(vib_index)
                                vib_type = "qrot"
                                line = line[:-1] + f"  # {vib_index:>3}{vib_type:>6}{mol.hinderedrotor._reduced_moms[itemindex]:>9.4f}(from G16){hindrot_item_reduced_mominert_dict[dummy_name][itemindex]:>9.4f}(from Mominert)   {mol.hinderedrotor._symmetry_numbers[itemindex]}   1"
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
    command = f"cd {thermo_path}; " + config.machine.thermo_command
    subprocess.call(command, shell=True)
