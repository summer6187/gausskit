import os
import copy
import subprocess
from pathlib import Path
import numpy as np

from gausskit.multiwell.thermo import get_thermo_lines, get_thermo_head_lines
from gausskit.settings import Configuration

config = Configuration()

module = "[Ktools]"


def get_ktools_head_lines(
    title, temp, rcnt, ntts, pcnt, whatdo="nosavefiles",
    emax=40000, egrain=2.5, jmax=500, jgrain=1,
    imax1=501, isize=1002, emax2=20000.0,
):
    """Build the 9-line KTOOLS control header (order per ktools read_input.f).

    NOTE: this is DISTINCT from the 4-line THERMO header. KTOOLS requires
    title / 'KCAL MCC' / whatdo / Emax Egrain / Jmax Jgrain /
    Imax1 Isize Emax2 / Nt / temperatures / 'Nreac Ntts Nprod'.
    """
    nt = len(temp.split())
    temps = " ".join(t if "." in t else t + "." for t in temp.split())
    return [
        f"{title}",
        "KCAL   MCC",
        f"{whatdo}",
        f"{emax} {egrain}",
        f"{jmax} {jgrain}",
        f"{imax1}  {isize}   {emax2}",
        f"{nt}",
        f"{temps}",
        f"{rcnt} {ntts} {pcnt}",
    ]


def write_ktools(
    PES_data,
    thermo_methods,
    thermo_path:Path,
    hindrot_item_reduced_mominert_dict,
    datfile:Path = Path("ktools.dat"),
    verbose:bool = False,
):
    """Write an input file for the ``ktools`` program.

    Args:
        PES_data (dict): Parsed potential energy surface data.
        thermo_methods (dict): Thermo configuration dictionary.
        thermo_path (Path): Directory for generated files.
        hindrot_item_reduced_mominert_dict (dict): Reduced moments for hindered
            rotors.
        datfile (Path, optional): Output filename. Defaults to ``ktools.dat``.
        verbose (bool, optional): Enable progress messages. Defaults to
            ``False``.

    Note:
        ``thermo_methods`` carries two reaction-coordinate fields with *different*
        length conventions: ``rc_distances`` has one entry per PES surface
        (reac + every trial TS + prod) because it is indexed per group, while
        ``rc_modes`` has one entry per trial TS only.

    Returns:
        None
    """

    # Parse thermo_methods information
    rc_distances = thermo_methods["rc_distances"]
    rc_modes = thermo_methods["rc_modes"]
    if_tunneling = thermo_methods["tunneling"]
    if_hinderedrotor = thermo_methods["hinderedrotor"]
    if_anharm = thermo_methods["anharm"]
    thermo_methods["adj_barrier"]
    thermo_temp = thermo_methods["temperatures"]
    thermo_pressure = thermo_methods["pressures"]

    # remove the reaction-coordinate mode from each trial transition state.
    # If rc_modes is not given in PES.in, default to dropping mode 0 (the
    # lowest-frequency mode) at every trial surface.
    trial_items = [n for n in PES_data if str.isnumeric(n)]
    if not rc_modes:
        rc_modes = [0] * len(trial_items)
        print("[Ktools] rc_modes not set; defaulting to drop mode 0 (lowest) at each trial TS")
    if len(rc_modes) != len(trial_items):
        print("The number of rc_modes doesn't match trial items!")
        print("rc_modes won't be used!")
    else:
        for PES_num, rc_mode in zip(trial_items, rc_modes):
            for item in PES_data[PES_num]["PES_items"]:
                # deep-copy before truncating: PES_data["Mol"] is a direct reference
                # to the database Molecule, so mutating it in place would corrupt a
                # subsequent [Thermo]/[Multiwell] run on the same species/section.
                Mol = copy.deepcopy(PES_data[PES_num]["PES_items"][item]["Mol"])
                print(f"Remove {rc_mode} in {item} {Mol.frequencies[rc_mode]}")
                Mol.frequencies = np.delete(Mol.frequencies, rc_mode)
                PES_data[PES_num]["PES_items"][item]["Mol"] = Mol

    # rc_distances is indexed per PES group (reac + every trial TS + prod), so it must
    # carry one entry per surface. Guard explicitly instead of letting rc_distances[n]
    # raise a bare IndexError mid-write (rc_modes, by contrast, is per trial TS only).
    if not rc_distances:
        raise ValueError(
            "[Ktools] rc_distances (reaction_coordinate_distances) is required: "
            "one distance per surface (reac + trial TSs + prod)"
        )
    if len(rc_distances) != len(PES_data):
        raise ValueError(
            f"[Ktools] rc_distances has {len(rc_distances)} entries but the PES has "
            f"{len(PES_data)} surfaces (reac + trial TSs + prod); supply one distance "
            "per surface (rc_modes is separate: one per trial TS only)"
        )

    if "default" in thermo_temp:
        temp = "200 300 400 500 600 800 1000 1200 1400 1600 1800 2000"
    else:
        temp = thermo_temp
    if "default" in thermo_pressure:
        pass
    else:
        pass

    # gather PES_info
    # item_list: Mol1, TS2, Mol3
    item_list = []
    # item_mol_name_list: HCFC133a, HCFC133a-OH_ts, Radical133a
    item_mol_name_list = []
    # item_mol_type: reac, None, None, prod
    item_mol_type = []
    # item_barrier
    item_barrier = []
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
                mol_type = "ctst"
            item_mol_type.append(mol_type)
            # ktools SUMS the per-fragment energies of a multi-fragment block
            # (e.g. prod: A + B). Split the block energy across its fragments so
            # they sum back to the true asymptote instead of doubling it.
            n_frag = len(PES_data[PES_num]["PES_items"])
            forwards_barrier = PES_data[PES_num]["PES_energy"] / n_frag
            item_barrier.append(f"{forwards_barrier:.4f}   {rc_distances[n]}")
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

    rcnt = item_mol_type.count("reac")
    pcnt = item_mol_type.count("prod")
    ntts = len(item_mol_type) - rcnt - pcnt
    reaction_lines = get_ktools_head_lines(
        "ktools input generated by gausskit", temp, rcnt, ntts, pcnt
    )

    for n, (dummy_name, mol_type, Mol, barrier) in enumerate(zip(item_list, item_mol_type, item_Mol_list, item_barrier)):
        mol = Mol

        lines = get_thermo_lines(
            mol,
            dummy_name = dummy_name,
            thermo_path = thermo_path,
            mol_type = mol_type,
            forwards_barrier = barrier,
            backwards_barrier = 0,
            if_ktools = True,
            if_tunneling = if_tunneling,
            if_anharm = if_anharm,
            if_hinderedrotor = if_hinderedrotor,
            hindrot_item_Mol_dict = hindrot_item_Mol_dict,
            hindrot_item_reduced_mominert_dict = hindrot_item_reduced_mominert_dict,
        )

        reaction_lines += lines

    reaction_lines.append(f"  {os.linesep}")

    # prepare thermo.dat
    with open(datfile, "w") as f:
        for line in reaction_lines:
            f.write(f"{line} {os.linesep}")
        f.write(f"  {os.linesep}")


def run_ktools(
    datfile:Path = Path("ktools.dat"),
    verbose:bool = False,
):
    """Execute the ``ktools`` program using ``datfile`` as input.

    ``ktools`` takes the input file name as a command-line argument and writes
    its outputs (``<base>.canonical``, ``.veff``, ``.log`` ...) next to it, so
    there is no output file to relocate.

    Args:
        datfile (Path): ``ktools`` input file.
        verbose (bool, optional): Display the executed command. Defaults to
            ``False``.

    Returns:
        None
    """

    cwd = datfile.parent.absolute()

    command = f"cd {cwd}; " + config.machine.ktools_command + f" {datfile.name}"
    if verbose:
        print(f"{module:11} Run command: {command}")
    subprocess.call(command, shell=True)
