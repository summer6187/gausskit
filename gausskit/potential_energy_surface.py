import argparse
from pathlib import Path
import configparser
import ast

from ase.units import Hartree, kcal, mol
from gausskit.database import load_database
from gausskit.multiwell.workflow import run_thermo_workflow, run_multiwell_workflow
from gausskit._defaults import trail_line


def config_section_map(config, section):
    dict1 = {}
    options = config.options(section)
    for option in options:
        try:
            dict1[option.lower()] = config.get(section, option)
            if dict1[option.lower()] == -1:
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option.lower()] = None
    return dict1


def config_getboolean(config, section, option):
    section_map = config_section_map(config, section)
    if option.lower() in section_map:
        _bool = config.getboolean(section, option)
    else:
        _bool = False
    return _bool


def match_method(item_method_list, method):
    matched_method = []
    for item_method in item_method_list:
        if item_method == None:
            item_method = ""
        if method.lower() in item_method.lower():
            matched_method.append(item_method)
    # raise error
    if matched_method == []:
        print(f"No method matched for {method} in {item_method_list}!")
        print("Please check your database!")
    return matched_method


def get_item_energy(item, PES_method_dict):
    Eele_method = PES_method_dict["Eele_method"]
    ZPE_method = PES_method_dict["ZPE_method"]
    anharm_method = PES_method_dict["anharm_method"]

    # match method in item_method list
    item_method_list = item.keys()
    Eele_methods = match_method(item_method_list, Eele_method)
    Eele_method = Eele_methods[0]

    # get energies
    Eele = item[Eele_method].electronic_energy

    # get ZPE
    ZPE_methods = match_method(item_method_list, ZPE_method)

    # don't use hindrot calc for general ZPE Mols
    for _method in ZPE_methods:
        if _method.endswith("hindrot"):
            pass
        else:
            ZPE_method = _method

    ZPE_Mol = item[ZPE_method]
    # single atom have no ZPE
    if len(item[Eele_method].get_chemical_symbols()) == 1:
        ZPE = 0
    else:
        # if not single atom parse ZPE
        if anharm_method:
            ZPE = item[ZPE_method].anharm_zpe
        else:
            ZPE = item[ZPE_method].zpe
    E_0K = Eele + ZPE

    return E_0K, ZPE_Mol


def get_item_ts(item, PES_method_dict):
    ZPE_method = PES_method_dict["ZPE_method"]

    # match method in name_method list
    item_method_list = list(item.keys())
    ZPE_methods = match_method(item_method_list, ZPE_method)
    ZPE_method = ZPE_methods[0]

    # single atom have no ZPE
    if len(item[item_method_list[0]].get_chemical_symbols()) == 1:
        _ts = False
    else:
        _ts = item[ZPE_method].ts

    return _ts


def get_PES_data(database, PES_dict, PES_num_list, PES_methods, verbose=False):

    # init PES_data
    PES_data = {}
    # label for the dummy name of mols or TSs
    n_mol = 1

    # parse every item in this PES
    for PES_num in PES_num_list:
        item_string = PES_dict[str(PES_num)]
        item_string = "+ " + item_string
        item_list = item_string.split()

        PES_item_dict = {}
        plus_minus = "+"
        final_ts = False
        for n, item in enumerate(item_list):
            if item == "+":
                plus_minus = "+"
            elif item == "-":
                plus_minus = "-"
            else:
                # when item is not + or -, then it's a Mol
                E_0K, ZPE_Mol = get_item_energy(database[item], PES_methods)
                if ZPE_Mol.ts:
                    dummy_name = f"TS{n_mol}"
                else:
                    dummy_name = f"Mol{n_mol}"
                n_mol += 1
                PES_item_dict[dummy_name] = {}
                PES_item_dict[dummy_name]["mol_name"] = item
                PES_item_dict[dummy_name]["plus_minus"] = plus_minus
                PES_item_dict[dummy_name]["Mol"] = ZPE_Mol

                # add hindered rotor mols if possible
                if any(
                    [
                        item_method.endswith("hindrot")
                        for item_method in database[item].keys()
                    ]
                ):
                    matched_methods = match_method(
                        database[item].keys(), PES_methods["ZPE_method"] + "_hindrot"
                    )
                    hindrot_method = matched_methods[0]
                    PES_item_dict[dummy_name]["Mol_hindrot"] = database[item][
                        hindrot_method
                    ]

                PES_item_dict[dummy_name]["item_energy"] = E_0K * Hartree / (kcal / mol)
                item_ts = get_item_ts(database[item], PES_methods)
                PES_item_dict[dummy_name]["ts"] = item_ts
                if item_ts == True:
                    final_ts = True

        # PES_items = collections.namedtuple("PES_items", PES_item_dict.keys())(**PES_item_dict)
        PES_items = PES_item_dict
        PES_data[str(PES_num)] = {}
        PES_data[str(PES_num)]["PES_items"] = PES_items
        PES_data[str(PES_num)]["final_ts"] = final_ts
        PES_energy = 0
        for n, item in enumerate(PES_data[str(PES_num)]["PES_items"]):
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
            PES_data[PES_num]["reverse"] = ts_energy - PES_data[PES_num]["PES_energy"]
            PES_data[PES_num]["reverse_ref"] = ts_n
        else:
            PES_data[PES_num]["reverse"] = 0
            PES_data[PES_num]["reverse_ref"] = PES_num
    if verbose:
        for PES_num in PES_data:
            item_string_list = []
            for PES_item in PES_data[PES_num]["PES_items"]:
                item_string_list.append(
                    PES_data[PES_num]["PES_items"][PES_item]["plus_minus"]
                )
                dummy_name = PES_item
                mol_name = PES_data[PES_num]["PES_items"][PES_item]["mol_name"]
                item_string_list.append(f"{mol_name}({dummy_name})")
            item_string = " ".join(item_string_list)[2:]  # omit first "+ " sign
            energy = PES_data[PES_num]["PES_energy"]
            reverse_energy = PES_data[PES_num]["reverse"]
            reverse_ref = PES_data[PES_num]["reverse_ref"]

            print(
                f"  {PES_num:>2}. {item_string:45} {energy:>8.3f}  {reverse_energy:>8.3f}  (ref:{reverse_ref})"
            )

    return PES_data


def thermo_method_warning(thermo_methods, PES_methods):
    if thermo_methods["anharm"]:
        if thermo_methods["tunneling"]:
            print("WARNING, Thermo anharm conflict with tunneling")
            print("Exiting program!")
            exit()
        if thermo_methods["hinderedrotor"]:
            print("WARNING, Thermo anharm conflict with thermo_hinderedrotor")
            print("Exiting program!")
            exit()
        if not PES_methods["anharm_method"]:
            print("PES(ZPE) does not include anharm")
            print("Exiting program!")
            exit()
    return


def PES_parser(config, dry:bool=False, verbose:bool=False):
    # set PES Method
    if "Method" not in config.sections():
        print("No Method section found!")
        exit()
    else:
        PES_method = config_section_map(config, "Method")
        Eele_method = PES_method["eele"]
        ZPE_method = PES_method["zpe"]
        anharm_method = config_getboolean(config, "Method", "anharm")
        # load database
        filename = Path(PES_method["database"])
        print(f"Loading database from {filename}")
        database = load_database(filename)

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
        thermo_list = list(Thermo_method["pes"].split())
        thermo_dir = Thermo_method["dir"]
        thermo_tunneling = config_getboolean(config, "Thermo", "tunneling")
        thermo_hinderedrotor = config_getboolean(config, "Thermo", "hinderedrotor")
        thermo_anharm = config_getboolean(config, "Thermo", "anharm")
        if "adj_barrier" in Thermo_method:
            thermo_adj_barrier = list(Thermo_method["adj_barrier"].split())
        else:
            thermo_adj_barrier = []
        thermo_temperatures = Thermo_method["temperatures"]
        thermo_pressures = Thermo_method["pressures"]

        thermo_methods = {
            "tunneling": thermo_tunneling,
            "hinderedrotor": thermo_hinderedrotor,
            "anharm": thermo_anharm,
            "adj_barrier": thermo_adj_barrier,
            "temperatures": thermo_temperatures,
            "pressures": thermo_pressures,
        }
        thermo_method_warning(thermo_methods, PES_methods)

    # set Multiwell Method
    calc_multiwell = False
    if "Multiwell" in config.sections():
        calc_multiwell = True
        Multiwell_method = config_section_map(config, "Multiwell")
        multiwell_pes = list(Multiwell_method["pes"].split())
        multiwell_dir = Multiwell_method["dir"]
        multiwell_temperature = float(Multiwell_method["temperature"])
        multiwell_pressures = list(Multiwell_method["pressures"].split())
        multiwell_wells = ast.literal_eval(Multiwell_method["wells"]) # read a list
        if type(multiwell_wells) is int:
            # here we make sure this is a list object
            multiwell_wells = [multiwell_wells]
        multiwell_channels = ast.literal_eval(Multiwell_method["channels"])
        multiwell_tunneling = config_getboolean(config, "Multiwell", "tunneling")
        multiwell_anharm = config_getboolean(config, "Multiwell", "anharm")

        # bimolecular reaction is optional
        if Multiwell_method.get("bimolecular_channel"):
            multiwell_bimolecular_channel = ast.literal_eval(Multiwell_method["bimolecular_channel"])
        else:
            multiwell_bimolecular_channel = None
        if Multiwell_method.get("bimolecular_concentrations"):
            multiwell_bimolecular_concentrations = ast.literal_eval(Multiwell_method["bimolecular_concentrations"]) # read a list
        else:
            multiwell_bimolecular_concentrations = None

        # trail line
        if Multiwell_method.get("trails"):
            multiwell_trail_line = Multiwell_method["trails"]
        else:
            multiwell_trail_line = trail_line


        if type(multiwell_wells) is int:
            # here we make sure this is a list object
            multiwell_wells = [multiwell_wells]

        multiwell_methods = {
            "temperature": multiwell_temperature,
            "pressures": multiwell_pressures,
            "wells": multiwell_wells,
            "channels": multiwell_channels,
            "tunneling": multiwell_tunneling,
            "anharm": multiwell_anharm,
            "bimolecular_channel": multiwell_bimolecular_channel,
            "bimolecular_concentrations": multiwell_bimolecular_concentrations,
            "trails": multiwell_trail_line,
        }

    PES_datasets = {}
    # parse the rest sections
    print(
        "      {:45} {:>8}  {:>8}".format(
            "reaction", "Fwd barr.", "bkw barr.(kcal/mol)"
        )
    )
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
            PES_data = get_PES_data(
                database, PES_dict, PES_num_list, PES_methods, verbose=True
            )
            PES_datasets[section] = PES_data

    # thermo calc
    if calc_thermo:
        print("-----------thermo calculation-----------")
        for _thermo_PES in thermo_list:
            PES_data = PES_datasets[_thermo_PES]
            thermo_path = Path(thermo_dir) / f"thermo_{_thermo_PES}"
            print("thermo calculation of", _thermo_PES)
            run_thermo_workflow(PES_data, thermo_methods, thermo_path, dry=dry, verbose=verbose)

    # multiwell calc
    if calc_multiwell:
        print("----------multiwell calculation----------")
        for _multiwell_PES in multiwell_pes:
            PES_data = PES_datasets[_multiwell_PES]
            multiwell_path = Path(multiwell_dir) / f"multiwell_{_multiwell_PES}"
            print("multiwell calculation of", _multiwell_PES)
            run_multiwell_workflow(PES_data, multiwell_methods, multiwell_path, dry=dry, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="PES.in file")
    args = parser.parse_args()

    config = configparser.ConfigParser()

    filename = args.file
    config.read(filename)
    PES_parser(config)
