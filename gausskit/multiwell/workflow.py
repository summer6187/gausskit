from pathlib import Path

from gausskit.multiwell.mominert import (
    read_mominert_out, run_mominert, write_mominert, get_rotor,
)
from gausskit.multiwell.sctst import (
    write_parsctst, write_bdens, run_parsctst, run_bdens, fix_crp_file,
    write_paradensum, run_paradensum,
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


def _echo(tag, msg):
    """Transparency line: what file was prepared / what command runs at this step."""
    print(f"  {tag:6} {msg}")


def _hrd_for(hrd_map, mol_name):
    """The declared fitted-mode list for a species (case-insensitive), or None."""
    if not hrd_map:
        return None
    entry = hrd_map.get(str(mol_name).lower())
    return entry["modes"] if entry else None


def run_PES_densdata(
    PES_data:dict,
    densdata_path:Path = Path("DensData"),
    thermo_temp:str = "200 300 400 500 600 800 1000 1200 1400 1600 1800 2000",
    Egrain:str = "10   3000   4000   50000",
    if_hinderedrotor:bool = False,
    if_anharm = False,
    dry:bool = False,
    verbose:bool = True,
    hrd_map:dict = None,
):
    """Prepare density of states data for a full PES.

    Args:
        PES_data (dict): Potential energy surface description.
        densdata_path (Path, optional): Working directory. Defaults to
            ``"DensData"``.
        thermo_temp (str, optional): Temperature grid string.
        Egrain (str, optional): Energy grain specification.
        if_hinderedrotor (bool, optional): Include hindered rotor treatment.
        if_anharm: ``False`` | ``True`` | ``"ts"`` (anharm for TS species only;
            wells/reactants stay harmonic).
        dry (bool, optional): Skip execution of external programs.
        verbose (bool, optional): Print progress. Defaults to ``True``.
        hrd_map (dict, optional): ``{species_lower: {"path", "modes"}}`` from a PES.in
            ``[HRD]`` section — scan-fitted Vhrd2 rotors emitted natively in the decks.

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
        mol_name = item_mol_name_list[n]
        hrd_modes = _hrd_for(hrd_map, mol_name)
        print(f"prep {mol_name} ({dummy_name}):")

        # 2.1 write and run mominert
        datfile = f"{dummy_name}.coords"
        outfile = f"{dummy_name}.coords.out"
        write_mominert(mol, datfile=densdata_path / datfile, verbose=verbose)
        _echo("write", f"{densdata_path / datfile} (mominert input)")
        _echo("exec", f"mominert {datfile} -> {outfile}")
        run_mominert(densdata_path / datfile, densdata_path / outfile, verbose=verbose)
        krot, ad_rot = get_rotor(densdata_path / outfile, verbose=verbose)
        mol.krotor = krot
        mol.ad_rotor = ad_rot
        _echo("read", f"{densdata_path / outfile} (Krot={krot:.4f}, ADrot={ad_rot:.4f} amu*A2)")

        # 2.2 write and run densum
        datfile = f"{dummy_name}.vib"
        outfile = f"{dummy_name}.dens"
        write_densum(
            mol,
            fname=dummy_name,
            Egrain=Egrain,
            datfile=densdata_path / datfile,
            verbose=verbose,
            hrd_modes=hrd_modes,
        )
        _echo("write", f"{densdata_path / datfile} (densum input"
              + (f", {len(hrd_modes)} HRD" if hrd_modes else "") + ")")
        if not dry:
            _echo("exec", f"densum {datfile} -> {outfile}")
            run_densum(densdata_path / datfile, densdata_path / outfile, verbose=verbose)
        else:
            _echo("skip", f"densum {datfile} [--dry]")

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
            hrd_modes=hrd_modes,
        )
        datfile = densdata_path.absolute() / f"{dummy_name}.therm"
        outfile = densdata_path.absolute() / f"{dummy_name}.therm.out"
        _echo("write", f"{datfile} (single-species thermo input)")
        if not dry:
            _echo("exec", f"thermo {datfile.name} -> {outfile.name}")
            run_thermo(datfile, outfile, verbose=verbose)
        else:
            _echo("skip", f"thermo {datfile.name} [--dry]")

        # 2.3.1 read Electronic partition function from thermo output files
        if not dry:
            qele = read_electronic_partition_function(outfile, verbose=verbose)
            mol.electronic_partition_function = qele


    # 3 run bdens and/or parsctst if anharm
    # prepare bdens.dat or parsctst.dat
    # we only have one parsctst mission, so only one set of forw. backw. barrier
    # if_anharm == "ts": SCTST (parsctst/crp) for the TS species only; wells/reactants
    # keep their harmonic blocks (no paradensum) -- the isolate-the-TS-treatment model.
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
                # parsctst; declared HRD modes leave the coupled X-matrix and are
                # emitted as separable Vhrd2 rotors.
                hrd_modes = _hrd_for(hrd_map, mol_name)
                datfile = densdata_path / f"{dummy_name}.parsctst.dat"
                barrier = [forwards_barrier, backwards_barrier]
                write_parsctst(
                    mol=mol,
                    fname=dummy_name,
                    barrier=barrier,
                    Egrain=Egrain,
                    datfile=datfile,
                    verbose=verbose,
                    hrd_modes=hrd_modes,
                )
                _echo("write", f"{datfile} (parsctst deck"
                      + (f", {len(hrd_modes)} separable HRD" if hrd_modes else "") + ")")
                if not dry:
                    _echo("exec", f"parsctst {datfile.name} -> {dummy_name}.crp/.qcrp")
                    run_parsctst(datfile, verbose)
                    fix_crp_file(densdata_path / f"{dummy_name}.crp")
                    fix_crp_file(densdata_path / f"{dummy_name}.qcrp")
                else:
                    _echo("skip", f"parsctst {datfile.name} [--dry]")

            elif if_anharm == "ts":
                _echo("skip", f"paradensum {dummy_name} (anharm: ts -- harmonic well/reactant)")

            else:
                # paradensum: parallel anharmonic density of states (replaces serial bdens)
                datfile = densdata_path / f"{dummy_name}.paradensum.dat"
                write_paradensum(mol, dummy_name, Egrain, datfile, verbose)
                _echo("write", f"{datfile} (paradensum input)")
                if not dry:
                    _echo("exec", f"paradensum {datfile.name}")
                    run_paradensum(datfile, verbose)
                else:
                    _echo("skip", f"paradensum {datfile.name} [--dry]")

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
    hrd_map:dict = None,
):
    """Run a full thermo calculation workflow.

    Args:
        PES_data (dict): Potential energy surface data.
        thermo_methods (dict): Thermo computation options.
        thermo_path (Path, optional): Working directory. Defaults to ``thermo``.
        Egrain (str, optional): Energy grain specification.
        dry (bool, optional): Skip execution of external programs.
        verbose (bool, optional): Verbose output. Defaults to ``False``.
        hrd_map (dict, optional): PES.in ``[HRD]``-declared rotors (see run_PES_densdata).

    Returns:
        None
    """

    if_hinderedrotor = thermo_methods["hinderedrotor"]
    if_anharm = thermo_methods["anharm"]
    print(f"== thermo workflow -> {thermo_path}/  (anharm={if_anharm}"
          + (f", HRD species: {', '.join(hrd_map)}" if hrd_map else "") + ") ==")

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
        hrd_map=hrd_map,
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
        hrd_map=hrd_map,
    )
    _echo("write", f"{thermo_path / datfile} (thermo reaction deck)")
    if not dry:
        _echo("exec", f"thermo {datfile} -> reaction.out")
        run_thermo(
            datfile=thermo_path.absolute() / datfile,
            verbose=verbose,
        )
    else:
        _echo("skip", f"thermo {datfile} [--dry]")

def run_multiwell_workflow(
    PES_data:dict,
    multiwell_methods:dict,
    multiwell_path:Path = Path("multiwell"),
    Egrain:str = "10   3000   4000   50000",
    dry:bool = False,
    verbose:bool = False,
    hrd_map:dict = None,
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
        hrd_map=hrd_map,
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

