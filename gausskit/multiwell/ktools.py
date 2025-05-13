import os
from pathlib import Path

from gausskit.multiwell.thermo import get_thermo_lines, get_thermo_head_lines

def write_ktools(
    PES_data,
    thermo_methods,
    thermo_path:Path,
    hindrot_item_reduced_mominert_dict,
    datfile:Path = Path("ktools.dat"),
    verbose:bool = False,
):

    # Parse thermo_methods information
    if_tunneling = thermo_methods["tunneling"]
    if_hinderedrotor = thermo_methods["hinderedrotor"]
    if_anharm = thermo_methods["anharm"]
    thermo_adj_barrier = thermo_methods["adj_barrier"]
    thermo_temp = thermo_methods["temperatures"]
    thermo_pressure = thermo_methods["pressures"]

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
    item_list = []
    # item_mol_name_list: HCFC133a, HCFC133a-OH_ts, Radical133a
    item_mol_name_list = []
    # item_mol_type: reac, None, None, prod
    item_mol_type = []
    # item_Mol_list: 3 Molecules objects
    item_Mol_list = []
    for n, PES_num in enumerate(PES_data):
        for item in PES_data[PES_num]["PES_items"]:
            item_list.append(item)
            item_mol_name_list.append(PES_data[PES_num]["PES_items"][item]["mol_name"])
            Mol = PES_data[PES_num]["PES_items"][item]["Mol"]
            item_Mol_list.append(Mol)
            if PES_num.lower() in ["reac", "prod"]:
                mol_type = PES_num.lower()
            else:
                mol_type = None
            item_mol_type.append(mol_type)
        if PES_data[PES_num]["final_ts"] == True:
            forwards_barrier = PES_data[PES_num]["PES_energy"]
            reverse_PES_num = list(PES_data.keys())[n + 1]
            backwards_barrier = PES_data[reverse_PES_num]["reverse"]
            # break

    # prepare hindered rot calculations
    # hindrot_item_Mol_dict: Molecule Objects if exists hindrot calculation
    hindrot_item_Mol_dict = {}
    if if_hinderedrotor:
        for n, PES_num in enumerate(PES_data):
            for item in PES_data[PES_num]["PES_items"]:
                if "Mol_hindrot" in PES_data[PES_num]["PES_items"][item]:
                    hindrot_item_Mol_dict[item] = PES_data[PES_num]["PES_items"][item][
                        "Mol_hindrot"
                    ]

    reaction_lines = get_thermo_head_lines(temp, len(item_list))

    for n, (dummy_name, mol_type, Mol) in enumerate(zip(item_list, item_mol_type, item_Mol_list)):
        mol = Mol

        lines = get_thermo_lines(
            mol,
            dummy_name = dummy_name,
            thermo_path = thermo_path,
            mol_type = mol_type,
            forwards_barrier = forwards_barrier,
            backwards_barrier = backwards_barrier,
            if_tunneling = if_tunneling,
            if_anharm = if_anharm,
            if_hinderedrotor = if_hinderedrotor,
            hindrot_item_Mol_dict = hindrot_item_Mol_dict,
        )

        reaction_lines += lines

    reaction_lines.append(f"  {os.linesep}")

    # prepare thermo.dat
    with open(datfile, "w") as f:
        for line in reaction_lines:
            f.write(f"{line} {os.linesep}")
        f.write(f"  {os.linesep}")
