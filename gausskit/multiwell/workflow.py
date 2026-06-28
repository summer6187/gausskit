from pathlib import Path

from gausskit.multiwell.mominert import (
    read_mominert_out, run_mominert, write_mominert, get_rotor,
)
from gausskit.multiwell.sctst import (
    write_parsctst, write_bdens, run_parsctst, run_bdens, fix_crp_file,
    write_paradensum, run_paradensum, _fix_paradensum_qvib,
    dos_engine_for_mol, paradensum_configured, n_vib_modes, DOS_WL_MIN_MODES,
)
from gausskit.multiwell.densum import write_densum, run_densum
from gausskit.multiwell.ktools import write_ktools, run_ktools
from gausskit.multiwell.thermo import (
    read_electronic_partition_function, write_thermo, write_single_thermo, run_thermo
)
from gausskit.multiwell.multiwell import (
    write_multiwell, run_multiwell, run_bimol_thermo
)

from gausskit.settings import Configuration

config = Configuration()

def run_PES_densdata(
    PES_data:dict,
    densdata_path:Path = Path("DensData"),
    thermo_temp:str = "200 300 400 500 600 800 1000 1200 1400 1600 1800 2000",
    Egrain:str = "10   3000   4000   50000",
    if_hinderedrotor:bool = False,
    if_anharm:bool = False,
    dry:bool = False,
    verbose:bool = True,
):
    """Prepare density of states data for a full PES.

    Args:
        PES_data (dict): Potential energy surface description.
        densdata_path (Path, optional): Working directory. Defaults to
            ``"DensData"``.
        thermo_temp (str, optional): Temperature grid string.
        Egrain (str, optional): Energy grain specification.
        if_hinderedrotor (bool, optional): Include hindered rotor treatment.
        if_anharm (bool, optional): Enable anharmonic calculations.
        dry (bool, optional): Skip execution of external programs.
        verbose (bool, optional): Print progress. Defaults to ``True``.

    Returns:
        dict: Mapping from dummy names to reduced moments when hindered rotors
        are processed.
    """

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
        if PES_data[PES_num]["final_ts"]:
            forwards_barrier = PES_data[PES_num]["PES_energy"]
            reverse_PES_num = list(PES_data.keys())[n + 1]
            backwards_barrier = PES_data[reverse_PES_num]["reverse"]

    print(item_mol_name_list)

    # 1. prepare working directory
    if not densdata_path.exists():
        densdata_path.mkdir(parents=True)

    # 2 write and run mominert and densum
    for n, (dummy_name, Mol) in enumerate(zip(item_list, item_Mol_list)):
        mol = Mol

        # 2.1 write and run mominert
        datfile = f"{dummy_name}.coords"
        outfile = f"{dummy_name}.coords.out"
        write_mominert(mol, datfile=densdata_path / datfile, verbose=verbose)
        run_mominert(densdata_path / datfile, densdata_path / outfile, verbose=verbose)
        krot, ad_rot = get_rotor(densdata_path / outfile, verbose=verbose)
        mol.krotor = krot
        mol.ad_rotor = ad_rot

        # 2.2 write and run densum (harmonic DOS).
        # In anharmonic mode the anharm loop (step 3) computes the DOS for every
        # multi-atom species and would overwrite this harmonic .dens, so skip the
        # redundant densum there. Skipping it also means a pre-placed anharmonic
        # .dens is reused instead of being clobbered, and it stops a stray harmonic
        # .dens from making the anharm-loop skip-guard wrongly skip (which would
        # leave the bimolecular thermo without its .qvib). Single atoms (no
        # vibrations) still go through densum -- the anharm loop skips them.
        datfile = f"{dummy_name}.vib"
        outfile = f"{dummy_name}.dens"
        if not (if_anharm and len(mol.numbers) > 1):
            write_densum(
                mol,
                fname=dummy_name,
                Egrain=Egrain,
                datfile=densdata_path / datfile,
                verbose=verbose
            )
            if not dry:
                run_densum(densdata_path / datfile, densdata_path / outfile, verbose=verbose)

        # 2.3 write and run thermo file for each molecules
        if "default" in thermo_temp:
            temp = "200 300 400 500 600 800 1000 1200 1400 1600 1800 2000"
        else:
            temp = thermo_temp
        write_single_thermo(
            temp,
            mol,
            dummy_name,
            densdata_path,
            verbose=verbose,
        )
        datfile = densdata_path.absolute() / f"{dummy_name}.therm"
        outfile = densdata_path.absolute() / f"{dummy_name}.therm.out"
        if not dry:
            run_thermo(datfile, outfile, verbose=verbose)

        # 2.3.1 read Electronic partition function from thermo output files
        if not dry:
            qele = read_electronic_partition_function(outfile, verbose=verbose)
            mol.electronic_partition_function = qele


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
                write_parsctst(
                    mol=mol,
                    fname=dummy_name,
                    barrier=barrier,
                    Egrain=Egrain,
                    datfile=datfile,
                    verbose=verbose
                )
                # The SCTST cumulative reaction probability is barrier- and
                # concentration-independent; reuse a pre-placed .crp/.qcrp instead
                # of recomputing the (expensive) parsctst when both already exist.
                crp = densdata_path / f"{dummy_name}.crp"
                qcrp = densdata_path / f"{dummy_name}.qcrp"
                if not dry:
                    if crp.exists() and qcrp.exists():
                        print(f"[Multiwell] reuse existing CRP/QCRP for {dummy_name} (skip parsctst)")
                    else:
                        run_parsctst(datfile, verbose)
                        fix_crp_file(crp)
                        fix_crp_file(qcrp)

            else:
                # Anharmonic vibrational DOS. The engine is chosen by mode count
                # (dos_engine_for_mol): few-mode species (<= 5 modes / <= 3 atoms) use
                # bdens' EXACT direct count -- O(N(Emax)), trivial and crash-free here --
                # while larger species use paradensum's parallel Wang-Landau, whose cost is
                # ~linear in the energy range and independent of the (exponential) state
                # count. See the scaling note in sctst.py. The DOS is barrier-/
                # concentration-independent, so a pre-placed .dens is reused.
                dens = densdata_path / f"{dummy_name}.dens"
                if not dry:
                    if dens.exists():
                        print(f"[Multiwell] reuse existing DOS for {dummy_name} (skip)")
                    else:
                        engine = dos_engine_for_mol(mol)
                        # Fall back to (serial) bdens if paradensum is not configured in
                        # ~/.gausskitrc, so the pipeline still runs without an MPI build.
                        if engine == "paradensum" and not paradensum_configured():
                            print(f"[Multiwell] paradensum not configured -- falling back "
                                  f"to bdens for {dummy_name} ({n_vib_modes(mol)} modes; "
                                  f"serial Wang-Landau, may be slow)")
                            engine = "bdens"
                        if engine == "bdens":
                            datfile = densdata_path / f"{dummy_name}.bdens.dat"
                            # few-mode species: pure direct count (Eswitch = Emax). Large
                            # fallback species (>= DOS_WL_MIN_MODES): keep the hybrid switch
                            # (Wang-Landau above it), since pure direct count is infeasible.
                            emax = Egrain.split()[-1] if n_vib_modes(mol) < DOS_WL_MIN_MODES else None
                            write_bdens(mol, dummy_name, Egrain, datfile, verbose, eswitch=emax)
                            run_bdens(datfile, verbose)
                        else:
                            datfile = densdata_path / f"{dummy_name}.paradensum.dat"
                            write_paradensum(mol, dummy_name, Egrain, datfile, verbose)
                            run_paradensum(datfile, verbose)
                            # paradensum's .qvib needs the THERMO-layout patch before the
                            # bimolecular thermo step can read it (anharmonic partition fn).
                            qvib = densdata_path / f"{dummy_name}.qvib"
                            if qvib.exists():
                                _fix_paradensum_qvib(qvib)

    # 3.1 internal hindered rotor
    # prepare hindered rot calculations
    # hindrot_item_Mol_dict: Molecule Objects if exists hindrot calculation
    hindrot_item_reduced_mominert_dict = {}
    if if_hinderedrotor:
        hindrot_item_Mol_dict = {}
        for n, PES_num in enumerate(PES_data):
            for item in PES_data[PES_num]["PES_items"]:
                if "Mol_hindrot" in PES_data[PES_num]["PES_items"][item]:
                    # the populated rotor data lives on Mol_hindrot (the molecule
                    # parsed from the hindered-rotor scan); the plain "Mol" has an
                    # empty default Hinderedrotor.
                    mol_hindrot = PES_data[PES_num]["PES_items"][item]["Mol_hindrot"]
                    hindrot_item_Mol_dict[item] = mol_hindrot
                    dummy_name = item

                    # extra mominert pass for the internal rotor; use distinct
                    # filenames so we do not overwrite the principal-moment
                    # output produced by the first mominert pass above.
                    datfile = f"{dummy_name}.hindrot.coords"
                    outfile = f"{dummy_name}.hindrot.coords.out"
                    write_mominert(
                        mol_hindrot,
                        mol_hindrot.hinderedrotor,
                        densdata_path / datfile,
                        verbose=verbose,
                    )
                    if not dry:
                        run_mominert(densdata_path / datfile, densdata_path / outfile, verbose=verbose)
                        # read reduced moment of inertia from the .out file
                        _, _, _, reduced_moment_of_inertia = read_mominert_out(
                            densdata_path / outfile, verbose=verbose
                        )
                        hindrot_item_reduced_mominert_dict[item] = reduced_moment_of_inertia

    return hindrot_item_reduced_mominert_dict

def run_ktools_workflow(
    PES_data:dict,
    ktools_methods:dict,
    ktools_path:Path = Path("ktools"),
    Egrain:str = "10   3000   4000   50000",
    dry:bool = False,
    verbose:bool = False,
):
    """Run the workflow required to generate input for ``ktools``.

    Args:
        PES_data (dict): Potential energy surface data.
        ktools_methods (dict): Settings for the calculation.
        ktools_path (Path, optional): Working directory. Defaults to ``ktools``.
        Egrain (str, optional): Energy grain specification.
        dry (bool, optional): Do not execute external programs.
        verbose (bool, optional): Print progress. Defaults to ``False``.

    Returns:
        None
    """

    # run density of state data
    hindrot_item_reduced_mominert_dict = run_PES_densdata(
        PES_data,
        densdata_path=ktools_path.absolute(),
        thermo_temp=ktools_methods["temperatures"],
        Egrain=Egrain,
        if_anharm=False,
        dry=dry,
        verbose=verbose,
    )

    # prepare thermo input file reaction.dat
    datfile = "ktools.dat"
    write_ktools(
        PES_data,
        ktools_methods,
        thermo_path=ktools_path.absolute(),
        hindrot_item_reduced_mominert_dict=hindrot_item_reduced_mominert_dict,
        datfile=ktools_path.absolute() / datfile,
        verbose=verbose,
    )
    if not dry:
        run_ktools(
            datfile=ktools_path.absolute() / datfile,
            verbose=verbose,
        )

def run_thermo_workflow(
    PES_data:dict,
    thermo_methods:dict,
    thermo_path:Path = Path("thermo"),
    Egrain:str = "10   3000   4000   50000",
    dry:bool = False,
    verbose:bool = False,
):
    """Run a full thermo calculation workflow.

    Args:
        PES_data (dict): Potential energy surface data.
        thermo_methods (dict): Thermo computation options.
        thermo_path (Path, optional): Working directory. Defaults to ``thermo``.
        Egrain (str, optional): Energy grain specification.
        dry (bool, optional): Skip execution of external programs.
        verbose (bool, optional): Verbose output. Defaults to ``False``.

    Returns:
        None
    """

    if_hinderedrotor = thermo_methods["hinderedrotor"]
    if_anharm = thermo_methods["anharm"]

    # run density of state data
    hindrot_item_reduced_mominert_dict = run_PES_densdata(
        PES_data,
        densdata_path=thermo_path.absolute(),
        thermo_temp=thermo_methods["temperatures"],
        Egrain=Egrain,
        if_hinderedrotor=if_hinderedrotor,
        if_anharm=if_anharm,
        dry=dry,
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
    if not dry:
        run_thermo(
            datfile=thermo_path.absolute() / datfile,
            verbose=verbose,
        )

def run_multiwell_workflow(
    PES_data:dict,
    multiwell_methods:dict,
    multiwell_path:Path = Path("multiwell"),
    Egrain:str = "10   3000   4000   50000",
    dry:bool = False,
    verbose:bool = False,
):
    """Run the workflow for MultiWell master equation calculations.

    Args:
        PES_data (dict): Potential energy surface data.
        multiwell_methods (dict): MultiWell options.
        multiwell_path (Path, optional): Working directory. Defaults to
            ``multiwell``.
        Egrain (str, optional): Energy grain specification.
        dry (bool, optional): Skip running external programs.
        verbose (bool, optional): Verbose output. Defaults to ``False``.

    Returns:
        None
    """

    if_hinderedrotor = False # multiwell_methods["thermo_hinderedrotor"]
    if_anharm = multiwell_methods["anharm"]

    densdata_path = multiwell_path.absolute() / "DensData"

    # run density of state data
    run_PES_densdata(
        PES_data,
        densdata_path,
        Egrain=Egrain,
        if_hinderedrotor=if_hinderedrotor,
        if_anharm=if_anharm,
        dry=dry,
        verbose=verbose,
    )

    # run thermo for bimolecular reaction if required
    if multiwell_methods.get("bimolecular_channel"):
        run_bimol_thermo(
            PES_data,
            multiwell_methods,
            thermo_path=densdata_path,
            verbose=verbose,
        )

    # prepare thermo input file reaction.dat
    if multiwell_methods.get("bimolecular_concentrations"):
        for nn, concentration in enumerate(multiwell_methods["bimolecular_concentrations"]):
            datfile = f"multiwell_{nn}.dat"
            write_multiwell(
                PES_data,
                multiwell_methods,
                datfile=multiwell_path.absolute() / datfile,
                bimolecular_concentration=concentration,
                verbose=verbose,
            )

            run_multiwell(
                datfile=multiwell_path.absolute() / datfile,
                verbose=verbose,
            )


    else:
        datfile = "multiwell.dat"
        write_multiwell(
            PES_data,
            multiwell_methods,
            datfile=multiwell_path.absolute() / datfile,
            verbose=verbose,
        )

        run_multiwell(
            datfile=multiwell_path.absolute() / datfile,
            verbose=verbose,
        )

