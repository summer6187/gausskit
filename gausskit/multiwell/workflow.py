from pathlib import Path

from gausskit.multiwell.mominert import (
    read_mominert_out, run_mominert, write_mominert, get_rotor,
)
from gausskit.multiwell.densum import write_densum, run_densum
from gausskit.multiwell.thermo import write_thermo, run_thermo
from gausskit.multiwell.sctst import (
    write_parsctst, write_bdens, run_parsctst, run_bdens, fix_crp_file,
)

from gausskit.settings import Configuration

config = Configuration()

def run_PES_densdata(
    PES_data:dict,
    densdata_path:Path = Path("DensData"),
    Egrain:str="10   3000   4000   50000",
    if_hinderedrotor:bool = False,
    if_anharm:bool = False,
    verbose:bool = True,
):
    # gather PES_info
    # item_list: Mol1, TS2, Mol3
    item_list = []
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
            reverse_PES_num = list(PES_data.keys())[n + 1]
            backwards_barrier = PES_data[reverse_PES_num]["reverse"]
            break

    print(item_mol_name_list)

    # 1. prepare working directory
    if not densdata_path.exists():
        densdata_path.mkdir()

    # 2 write and run mominert and densum
    for n, (dummy_name, Mol) in enumerate(zip(item_list, item_Mol_list)):
        mol = Mol

        # 2.1 write and run mominert
        datfile = f"{dummy_name}.coords"
        outfile = f"{dummy_name}.coords.out"
        write_mominert(mol, datfile=densdata_path / datfile, verbose=verbose)
        run_mominert(densdata_path / datfile, densdata_path / outfile, verbose=verbose)
        krot, ad_rot = get_rotor(densdata_path / outfile, verbose=verbose)

        # 2.2 write and run densum
        datfile = f"{dummy_name}.vib"
        outfile = f"{dummy_name}.dens"
        write_densum(
            mol,
            krot,
            ad_rot,
            fname=dummy_name,
            Egrain=Egrain,
            datfile=densdata_path / datfile, 
            verbose=verbose
        )
        run_densum(densdata_path / datfile, densdata_path / outfile, verbose=verbose)


    # 3 run bdens and/or parsctst if anharm
    # prepare bdens.dat or parsctst.dat
    # we only have one parsctst mission, so only one set of forw. backw. barrier
    if if_anharm:
        if verbose:
            print("---------------anharmonic(sctst)---------------")
        for n, (dummy_name, Mol) in enumerate(zip(item_list, item_Mol_list)):

            mol = Mol
            mol_name = item_mol_name_list[n]

            if len(mol.numbers) <= 1:
                print(f"Skip this one atom file {mol_name}")
                continue

            if mol.ts:
                # parsctst
                datfile = densdata_path / f"{dummy_name}.parsctst.dat"
                barrier = [forwards_barrier, backwards_barrier]
                write_parsctst(mol, dummy_name, barrier, Egrain, datfile, verbose)
                run_parsctst(datfile, verbose)
                fix_crp_file(densdata_path / f"{dummy_name}.crp")
                fix_crp_file(densdata_path / f"{dummy_name}.qcrp")

            else:
                # bdens
                datfile = densdata_path / f"{dummy_name}.bdens.dat"
                write_bdens(mol, dummy_name, Egrain, datfile, verbose)
                run_bdens(datfile, verbose)

    # 3.1 internal hindered rotor
    # prepare hindered rot calculations
    # hindrot_item_Mol_dict: Molecule Objects if exists hindrot calculation
    hindrot_item_reduced_mominert_dict = {}
    if if_hinderedrotor:
        hindrot_item_Mol_dict = {}
        for n, PES_num in enumerate(PES_data):
            for item in PES_data[PES_num]["PES_items"]:
                if "Mol_hindrot" in PES_data[PES_num]["PES_items"][item]:
                    hindrot_item_Mol_dict[item] = PES_data[PES_num]["PES_items"][item][
                        "Mol_hindrot"
                    ]
                    mol = PES_data[PES_num]["PES_items"][item]["Mol"]
                    mol.hinderedrotor
                    dummy_name = item

                    datfile = f"{dummy_name}.coords"
                    outfile = f"{dummy_name}.coords.out"
                    write_mominert(mol, mol.hinderedrotor, densdata_path / datfile, verbose=verbose)
                    run_mominert(densdata_path / datfile, densdata_path / outfile, verbose=verbose)
                    # read moment of inertia from fname.coords.out files
                    _, _, _, reduced_moment_of_inertia = read_mominert_out(densdata_path / outfile, verbose=verbose)
                    hindrot_item_reduced_mominert_dict[item] = reduced_moment_of_inertia

    return hindrot_item_reduced_mominert_dict

def run_thermo_workflow(
    PES_data:dict,
    thermo_methods:dict,
    thermo_path:Path = Path("thermo"),
    Egrain:str = "10   3000   4000   50000",
    verbose:bool = False,
):

    if_hinderedrotor = thermo_methods["thermo_hinderedrotor"]
    if_anharm = thermo_methods["thermo_anharm"]

    # run density of state data
    hindrot_item_reduced_mominert_dict = run_PES_densdata(
        PES_data,
        densdata_path=thermo_path.absolute(),
        Egrain=Egrain,
        if_hinderedrotor=if_hinderedrotor,
        if_anharm=if_anharm,
        verbose=verbose,
    )

    # prepare thermo input file reaction.dat
    datfile = "reaction.dat"
    write_thermo(
        PES_data,
        thermo_methods,
        thermo_path=thermo_path.absolute(),
        hindrot_item_reduced_mominert_dict=hindrot_item_reduced_mominert_dict,
        datfile=thermo_path.absolute() / datfile,
        verbose=verbose,
    )

    run_thermo(
        datfile=thermo_path.absolute() / datfile,
        verbose=verbose,
    )

def run_multiwell_workflow(
    PES_data:dict,
    multiwell_methods:dict,
    multiwell_path:Path = Path("thermo"),
    Egrain:str = "10   3000   4000   50000",
    verbose:bool = False,
):
    return

    # if_hinderedrotor = thermo_methods["thermo_hinderedrotor"]
    # if_anharm = thermo_methods["thermo_anharm"]
    #
    # # run density of state data
    # hindrot_item_reduced_mominert_dict = run_PES_densdata(
    #     PES_data,
    #     densdata_path=thermo_path.absolute(),
    #     Egrain=Egrain,
    #     if_hinderedrotor=if_hinderedrotor,
    #     if_anharm=if_anharm,
    #     verbose=verbose,
    # )
    #
    # # prepare thermo input file reaction.dat
    # datfile = "reaction.dat"
    # write_thermo(
    #     PES_data,
    #     thermo_methods,
    #     thermo_path=thermo_path.absolute(),
    #     hindrot_item_reduced_mominert_dict=hindrot_item_reduced_mominert_dict,
    #     datfile=thermo_path.absolute() / datfile,
    #     verbose=verbose,
    # )
    #
    # run_thermo(
    #     datfile=thermo_path.absolute() / datfile,
    #     verbose=verbose,
    # )
    #
