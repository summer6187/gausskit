import sys, os
import configparser
import pickle
import numpy as np
from ase.units import Hartree, kcal, mol
from gausskit.multiwellkit.run_thermo import run_PES_thermo
import collections
import json


def config_section_map(config, section):
    dict1 = {}
    options = config.options(section)
    for option in options:
        try:
            dict1[option] = config.get(section, option)
            if dict1[option] == -1:
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1

def match_method(name_method_list, method):
    matched_method = ""
    for name_method in name_method_list:
        if name_method == None:
            name_method = ""
        if method.lower() in name_method.lower():
            matched_method = name_method
    return matched_method

def get_item_energy(name, PES_method_dict):
    Eele_method = PES_method_dict["Eele_method"]
    ZPE_method = PES_method_dict["ZPE_method"]
    anharm_method = PES_method_dict["anharm_method"]

    # match method in name_method list
    name_method_list = name.keys()
    Eele_method = match_method(name_method_list, Eele_method)

    # get energies
    Eele = name[Eele_method].electronic_energy

    # get ZPE
    ZPE_method = match_method(name_method_list, ZPE_method)
    ZPE_Mols = name[ZPE_method]
    # single atom have no ZPE
    if len(name[Eele_method].get_chemical_symbols()) == 1:
        ZPE = 0
    else:
        # if not single atom parse ZPE
        if anharm_method:
            ZPE = name[ZPE_method].anharm_zpe
        else:
            ZPE = name[ZPE_method].zpe
    
    E_0K = Eele + ZPE

    return E_0K, ZPE_Mols

def get_item_ts(name, PES_method_dict):
    ZPE_method = PES_method_dict["ZPE_method"]

    # match method in name_method list
    name_method_list = list(name.keys())
    ZPE_method = match_method(name_method_list, ZPE_method)


    # single atom have no ZPE
    if len(name[name_method_list[0]].get_chemical_symbols()) == 1:
        _ts = False
    else:
        _ts = name[ZPE_method].ts

    return _ts

def parse_this_PES(database, PES_dict, PES_num_list, PES_methods, verbose=False):
    
    # init PES_data
    PES_data = {}
    
    # parse every item in this PES
    for PES_num in PES_num_list:
        item_string = PES_dict[str(PES_num)]
        item_string = "+ " + item_string
        item_list = item_string.split()

        PES_item_dict = {}
        plus_minus = "+"
        for n, item in enumerate(item_list):
            final_ts = False
            if item == "+":
                plus_minus = "+"
            elif item == "-":
                plus_minus = "-"
            else:
                PES_item_dict[item] = {}
                PES_item_dict[item]["plus_minus"] = plus_minus
                E_0K, ZPE_Mols = get_item_energy(database[item], PES_methods)
                PES_item_dict[item]["Mols"] = ZPE_Mols
                PES_item_dict[item]["item_energy"] = E_0K * Hartree/(kcal/mol)
                item_ts = get_item_ts(database[item], PES_methods)
                PES_item_dict[item]["ts"] = item_ts
                if item_ts == True:
                    final_ts = True

        # PES_items = collections.namedtuple("PES_items", PES_item_dict.keys())(**PES_item_dict)
        PES_items = PES_item_dict
        PES_data[str(PES_num)] = {}
        PES_data[str(PES_num)]["PES_items"] = PES_items
        PES_data[str(PES_num)]["final_ts"] = final_ts
        PES_energy = 0
        for n,item in enumerate(PES_data[str(PES_num)]["PES_items"]):            
            if PES_data[str(PES_num)]["PES_items"][item]["plus_minus"] == "+":
                PES_energy += PES_data[str(PES_num)]["PES_items"][item]["item_energy"]
            elif PES_data[str(PES_num)]["PES_items"][item]["plus_minus"] == "-":
                PES_energy -= PES_data[str(PES_num)]["PES_items"][item]["item_energy"]
            else:
                print(f"Warning! no plus_minus found for this item {item}")
                exit()
            
        PES_data[str(PES_num)]["PES_energy"] = PES_energy
        
    # get reverse energy
    _reverse = False
    for n, PES_num in enumerate(PES_data):
        if n == 0:
            PES_ref_energy = PES_data[PES_num]["PES_energy"]
        PES_data[PES_num]["PES_energy"] -= PES_ref_energy
        if PES_data[PES_num]["final_ts"]:
            PES_data[PES_num]["reverse"] = 0
            PES_data[PES_num]["reverse_ref"] = PES_num 
            ts_energy = PES_data[PES_num]["PES_energy"]
            ts_n = PES_num
            _reverse = True
        elif _reverse:
            PES_data[PES_num]["reverse"] = ts_energy-PES_data[PES_num]["PES_energy"]
            PES_data[PES_num]["reverse_ref"] = ts_n
        else:
            PES_data[PES_num]["reverse"] = 0
            PES_data[PES_num]["reverse_ref"] = PES_num
    if verbose:
        for PES_num in PES_data:
            item_string_list = []
            for PES_item in PES_data[PES_num]["PES_items"]:
                item_string_list.append(PES_data[PES_num]["PES_items"][PES_item]["plus_minus"])
                item_string_list.append(PES_item)
            item_string = " ".join(item_string_list)[2:] # omit first "+ " sign
            energy = PES_data[PES_num]["PES_energy"]
            reverse_energy = PES_data[PES_num]["reverse"]
            reverse_ref = PES_data[PES_num]["reverse_ref"]

            print(f"  {PES_num:>2}. {item_string:35} {energy:>8.3f}  {reverse_energy:>8.3f}  (backwards)  ref:{reverse_ref}")
    
    return PES_data


if __name__ == "__main__":
    config = configparser.ConfigParser()

    # filename = "testcases/PES.ini"
    filename = sys.argv[1]
    config.read(filename)

    # set PES Method
    if "Method" not in config.sections():
        print("No Method section found!")
        exit()
    else:
        PES_method = config_section_map(config, "Method")
        Eele_method = PES_method["eele"]
        ZPE_method = PES_method["zpe"]
        anharm_method = config.getboolean("Method", "anharm")
        # load database
        filename = PES_method["database"]
        with open(filename, "rb") as f:
            database = pickle.load(f)

    PES_methods = {
        "Eele_method": Eele_method,
        "ZPE_method": ZPE_method,
        "anharm_method": anharm_method,
    }

    # set Thermo Method
    calc_thermo = False
    if "Thermo" in config.sections():
        calc_thermo = True
        Thermo_method = config_section_map(config, "Thermo")
        thermo_list = list(Thermo_method["thermo"].split())
        thermo_dir = Thermo_method["thermo_dir"]
        thermo_tunneling = config.getboolean("Thermo", "tunneling")
        thermo_internal_rotor = config.getboolean("Thermo", "internal_rotor")
        
        thermo_methods = {
            "thermo_tunneling": thermo_tunneling,
            "thermo_internal_rotor": thermo_internal_rotor,
        }

    PES_datasets = {}
    # parse the rest sections
    for section in config.sections():
        # parse one PES
        if section != "Method" and "PES" in section:
            print(section)
            # init PES_datasets for each PES_n
            PES_datasets[section] = {}
            PES_dict = config_section_map(config, section)
            PES_num_list = PES_dict.keys()
            PES_num_list = [int(num) for num in PES_num_list]
            PES_num_list.sort()

            PES_data = parse_this_PES(database, PES_dict, PES_num_list, PES_methods, verbose=True)
            PES_datasets[section] = PES_data

    # thermo calc
    if calc_thermo:
        print("thermo calculation")
        for _thermo_PES in thermo_list:
            PES_data = PES_datasets[_thermo_PES]
            thermo_path = os.path.join(thermo_dir, "thermo_" + _thermo_PES)
            print("thermo calculation of", _thermo_PES)
            run_PES_thermo(PES_data, thermo_methods, thermo_path=thermo_path, verbose=True)

