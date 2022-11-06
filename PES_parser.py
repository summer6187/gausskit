import sys
import configparser
import pickle
import numpy as np
from ase.units import Hartree, kcal, mol


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

def get_PES_energy(name, PES_method_dict):
    Eele_method = PES_method_dict["Eele_method"]
    ZPE_method = PES_method_dict["ZPE_method"]
    anharm_method = PES_method_dict["anharm_method"]

    # match method in name_method list
    name_method_list = name.keys()
    Eele_method = match_method(name_method_list, Eele_method)

    # get energies
    Eele = name[Eele_method].electronic_energy

    # single atom have no ZPE
    if len(name[Eele_method].get_chemical_symbols()) == 1:
        ZPE = 0
    else:
        # if not single atom parse ZPE
        ZPE_method = match_method(name_method_list, ZPE_method)
        if anharm_method:
            ZPE = name[ZPE_method].anharm_zpe
        else:
            ZPE = name[ZPE_method].zpe
    
    E_0K = Eele + ZPE

    return E_0K

def get_PES_ts(name, PES_method_dict):
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
        anharm_method = bool(PES_method["anharm"])
        
        # load database
        filename = PES_method["database"]
        with open(filename, "rb") as f:
            database = pickle.load(f)


    PES_method_dict = {
        "Eele_method": Eele_method,
        "ZPE_method": ZPE_method,
        "anharm_method": anharm_method,
    }

    # parse the rest sections
    for section in config.sections():
        # parse one PES
        if section != "Method":
            print(section)
            PES_dict = config_section_map(config, section)
            PES_num_list = PES_dict.keys()
            PES_num_list = [int(num) for num in PES_num_list]
            PES_num_list.sort()

            name_list = []
            energy_list = []
            ts_list = []
            
            # parse every item in this PES
            for PES_num in PES_num_list:
                item_string = PES_dict[str(PES_num)]
                name_list.append(item_string)
                item_list = item_string.split()
                
                # parse the energy calculation of this item
                final_E = 0
                # if there is any ts in a line, then it is ts
                final_ts = []
                for n, item in enumerate(item_list):
                    # add the first
                    if n == 0:
                        final_E += get_PES_energy(database[item], PES_method_dict)
                        final_ts.append(get_PES_ts(database[item], PES_method_dict))
                    # parse the equation
                    if item == "+":
                        final_E += get_PES_energy(database[item_list[n+1]], PES_method_dict)
                    elif item == "-":
                        final_E -= get_PES_energy(database[item_list[n+1]], PES_method_dict)
                    else:
                        # if the item is not +/-, then parse the item and get if_ts
                        final_ts.append(get_PES_ts(database[item], PES_method_dict))
                energy_list.append(final_E)
                # if there is any ts in a line, then it is ts
                if True in final_ts:
                    ts_list.append(True)
                else:
                    ts_list.append(False)
            energy_list = np.array(energy_list)

            # get relative energy regarding to the first final_E
            energy_list = energy_list - energy_list[0]
            energy_list *= Hartree/(kcal/mol)
            
            # print the energy results
            print_reverse = False
            for n, (name, energy, ts) in enumerate(zip(name_list,energy_list,ts_list)):
                if ts:
                    print(f"  {n}. {name:35} {energy:.4f}                         (ts)")
                    ts_energy = energy
                    ts_n = n
                    print_reverse = True
                elif print_reverse:
                    print(f"  {n}. {name:35} {energy:.4f}   {ts_energy-energy:.4f}  (backwards)     (ref: {ts_n})")
                else:
                    print(f"  {n}. {name:35} {energy:.4f}")

