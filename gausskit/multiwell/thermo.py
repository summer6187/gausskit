from pathlib import Path
import os
import subprocess
from typing import Optional

import numpy as np

from gausskit.molecules import Molecules
from gausskit.multiwell.densum import get_degrees_of_freedom_lines
from gausskit.multiwell.mominert import get_rotor
from gausskit.settings import Configuration

config = Configuration()

module = "[Thermo]"

def get_thermo_head_lines(temp, n_species):
    """Return header lines for a ``thermo`` input file.

    Args:
        temp (str): Temperature grid specification.
        n_species (int): Number of species blocks to follow.

    Returns:
        list[str]: The header lines.
    """

    head_lines = [
        "KCAL   MCC",
        str(len(temp.split())),
        temp,
        f"{n_species}",
    ]

    return head_lines


def get_thermo_lines(
    mol:Molecules,
    dummy_name:str,
    thermo_path:Path,
    mol_type:Optional[str] = None,
    forwards_barrier:float = 0.0,
    backwards_barrier:float = 0.0,
    if_ktools:float = False,
    if_tunneling:bool = False,
    if_anharm = False,
    if_hinderedrotor:bool = False,
    hindrot_item_Mol_dict:dict = {},
    hindrot_item_reduced_mominert_dict:dict = {},
    hrd_modes:list = None,
):
    """Generate the block describing ``mol`` for a ``thermo`` input file.

    Args:
        mol (Molecules): Molecule being described.
        dummy_name (str): Identifier used in the input file.
        thermo_path (Path): Directory for auxiliary files.
        mol_type (str, optional): ``'reac'``, ``'prod'`` or ``'ctst'``.
        forwards_barrier (float, optional): Forward barrier height in kcal/mol.
        backwards_barrier (float, optional): Reverse barrier height.
        if_ktools (bool, optional): Use ktools formatting. Defaults to ``False``.
        if_tunneling (bool, optional): Include tunnelling information.
        if_anharm: ``False`` | ``True`` | ``"ts"`` — ``"ts"`` applies the anharmonic
            (crp/SCTST) block only when ``mol.ts``; other species stay harmonic.
        if_hinderedrotor (bool, optional): Include hindered rotors (legacy HINDROT path).
        hindrot_item_Mol_dict (dict, optional): Mapping of dummy names to
            hindered rotor molecules.
        hrd_modes (list, optional): PES.in ``[HRD]``-declared fitted rotors — the paired
            soft vib lines are emitted as hrd/Vhrd2/Bhrd1 blocks (harmonic path only).

    Returns:
        list[str]: Lines describing ``mol``.
    """
    lines = []
    # tri-state anharm: "ts" means the anharm (crp) block applies to the TS only
    anharm_here = if_anharm is True or (if_anharm == "ts" and mol.ts)

    if mol_type is not None:
        # ktools path passes forwards_barrier as a preformatted "energy   bond" string;
        # the thermo path passes a float. Handle both.
        bar = forwards_barrier if isinstance(forwards_barrier, str) else f"{forwards_barrier:.4f}"
        lines.append(f"{mol_type}    {dummy_name}    {bar}")
    else:
        if mol.ts:
            # if no tunneling, set img_freq and backwards_barrier to 0
            if not if_tunneling:
                img_freq = 0
                backwards_barrier = 0
            else:
                freq = mol.frequencies
                img_freq_list = freq[freq<0]
                assert len(img_freq_list) == 1, f"Something wrong with img frequency {img_freq_list}"
                img_freq = img_freq_list[0]
            lines.append(
                f"ctst    {dummy_name}    {forwards_barrier:.4f}   {-img_freq:.4f}   {backwards_barrier:.4f}"
            )
        else:
            lines.append(f"reac    {dummy_name}    {forwards_barrier:.4f}")

    lines.append(f"{mol.get_chemical_formula()}")
    if if_ktools:
        # exactly three comment lines: a species' provenance/reference lines (e.g. set
        # with electronic_states via the name(spin-orbit) tag) if present, else generic.
        comments = getattr(mol, "electronic_comments", None)
        if comments:
            for c in (list(comments) + ["", "", ""])[:3]:
                lines.append(c)
        else:
            lines.append("1. Comment line")
            lines.append("2. Comment line")
            lines.append("3. Comment line")
    # electronic states: a tabulated multi-level ladder (e.g. spin-orbit, set via the
    # name(spin-orbit) PES.in tag) if present, else the ground level only with
    # degeneracy = spin multiplicity (a Gaussian log carries nothing more -- see
    # gausskit.electronic_states).
    elec_states = getattr(mol, "electronic_states", None)
    if elec_states:
        lines.append(f"{mol.external_symmetry_number}   {mol.optical_isomers}   {len(elec_states)}")
        for energy, degeneracy in elec_states:
            lines.append(f" {energy:<10} {degeneracy}")
    else:
        lines.append(f"{mol.external_symmetry_number}   {mol.optical_isomers}   1")
        lines.append(f" {0.0:<10} {mol.multiplicity}")

    mominert_outfile = f"{dummy_name}.coords.out"
    krot, ad_rot = get_rotor(thermo_path / mominert_outfile)

    # if run anharmonic thermo
    if anharm_here:
        # count the number of k-rotor and adiabatic rotor [NOT hindered rotor!!!]
        total_dof = 1 # read external file, this takes "one vibration"
        if np.abs(krot) > 1e-12:
            total_dof += 1
        if np.abs(ad_rot) > 1e-12:
            total_dof += 1

        # in case of one-atom species, don't write any degrees of freedom
        if len(mol.numbers) <= 1:
            total_dof = 0
            lines.append(f"{total_dof}   HAR   AMUA")
            lines.append(" ")
            return lines

        # number of vibrations and rotations to be read in
        lines.append(f"{total_dof}   HAR   AMUA")

        # add read external file line
        if mol.ts:
            line = "1    crp   0.0      1.0     1     ! read external file"
        else:
            line = "1    qvb   0.0      1.0     1     ! read external file"
        lines.append(line)

        # add k-rotor and adiabatic rotor line
        # additional mode number begins with 2
        if if_ktools:
            rottype = "kro"
        else:
            rottype = "qrot " if (krot < 11.0) else "rot  "
        n_dof = 1
        if np.abs(krot) > 1e-12:
            n_dof += 1
            lines.append(
                f" {n_dof:3d}   {rottype:6} {krot:12.4f}   1.0   1   ! K-rotor"
            )

        if if_ktools:
            rottype = "jro"
        else:
            rottype = "qrot " if (ad_rot < 11.0) else "rot  "
        if np.abs(ad_rot) > 1e-12:
            n_dof += 1
            lines.append(
                f" {n_dof:3d}   {rottype:6} {ad_rot:12.4f}   1.0   2   ! 2D adiabatic rotor"
            )

    # with normal modes, lines will be the same as fname.vibs
    else:
        # get all non imaginary frequencies
        freq = mol.frequencies
        nonimg_freq = freq[freq>0]
        total_dof = len(nonimg_freq)
        if np.abs(krot) > 1e-12:
            total_dof += 1
        if np.abs(ad_rot) > 1e-12:
            total_dof += 1
        lines.append(f"{total_dof}   HAR   AMUA")
        dof_lines = get_degrees_of_freedom_lines(mol, krot, ad_rot, if_ktools=if_ktools,
                                                 hrd_modes=hrd_modes)

        # if thermo_hinderedrotor and mol has hindered rotor
        # replace the selected vibration mode with the hindered rotor DOF
        if if_hinderedrotor and dummy_name in hindrot_item_Mol_dict:
            # vibration in xxx.therm file is One-based numbering
            mol = hindrot_item_Mol_dict[dummy_name]
            corrected_vibs = mol.hinderedrotor._corrected_vibs
            for n_index, n_vib in enumerate(corrected_vibs):
                corr_rot = mol.hinderedrotor._reduced_moms[n_index]
                rottype = "qrot " if (corr_rot < 11.0) else "rot  "
                line = f"  # {n_vib:>3}{rottype:>6}{corr_rot:>9.4f}(from G16)"
                line += f"{hindrot_item_reduced_mominert_dict[dummy_name][n_index]:>9.4f}(from Mominert)  " # FIXME: wtf???
                line += f" {mol.hinderedrotor._symmetry_numbers[n_index]}   1"
                dof_lines[n_vib] = line

        lines += dof_lines
    lines.append(f"  {os.linesep}")
    return lines


def parse_adj_barrier(adj_barrier):
    """Parse the ``adj_barrier`` spec into per-species barrier overrides.

    ``adj_barrier`` is the white-space-split list read from the INI
    (``[Thermo]``/``[Multiwell]`` key ``adj_barrier``). Every entry MUST name the
    species explicitly as ``NAME=VALUE`` -- a bare value is rejected so the code
    never has to guess which transition state to adjust:

    * ``NAME``  -- the channel/TS to adjust, given either by its full name as
      written in the ``[PES...]`` section (e.g. ``Alkoxyl133a-HLeave_O2_another_ts``)
      or by its short deck name (e.g. ``TS16``).
    * ``VALUE`` -- the new *absolute* forward-barrier height in kcal/mol.

    Any number of channels may be adjusted in one line, e.g.::

        adj_barrier: Alkoxyl133a-HLeave_O2_another_ts=1.85 Alkoxyl133a-RLeave_ts=10.0

    (entries may be separated by white space and/or commas.)

    Returns:
        dict: ``{name: barrier}`` (absolute kcal/mol).

    Raises:
        ValueError: an entry is not in ``NAME=VALUE`` form (e.g. a bare value).
    """
    overrides = {}
    for token in adj_barrier or []:
        token = token.strip().rstrip(",").strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"adj_barrier entry '{token}' must name the species as NAME=VALUE "
                f"(e.g. 'Alkoxyl133a-RLeave_ts=10.0'); a bare value is not allowed -- "
                f"name the channel explicitly so the TS to adjust is unambiguous."
            )
        name, _, value = token.partition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"adj_barrier entry '{token}' has an empty species name.")
        overrides[name] = float(value)
    return overrides


def apply_adj_barrier(item_list, item_mol_name_list, item_Mol_list,
                      forwards_barrier_list, adj_barrier, verbose=False):
    """Override forward barriers per the ``adj_barrier`` spec (any named channel).

    Each ``NAME=VALUE`` replaces the forward barrier of the matching channel,
    matched against EITHER its short deck name (``item_list``, e.g. ``TS16``) OR
    its full ``[PES...]`` name (``item_mol_name_list``). Any channel can be
    adjusted, not only a unique TS. A no-op when ``adj_barrier`` is empty.

    Returns a new ``forwards_barrier_list``.

    Raises:
        ValueError: a named species matches nothing in this reaction.
    """
    if not adj_barrier:
        return forwards_barrier_list

    overrides = parse_adj_barrier(adj_barrier)

    new_list = list(forwards_barrier_list)
    matched = set()
    for i, (dummy, molname, mol, old) in enumerate(
        zip(item_list, item_mol_name_list, item_Mol_list, forwards_barrier_list)
    ):
        key = dummy if dummy in overrides else (molname if molname in overrides else None)
        if key is None:
            continue
        new_list[i] = overrides[key]
        matched.add(key)
        if not getattr(mol, "ts", False):
            print(f"{module:10} Warning: adj_barrier target '{key}' is not a "
                  f"transition state; adjusting its barrier anyway")
        print(f"{module:10} adj_barrier: {key} ({dummy}) forward barrier "
              f"{old:.4f} -> {overrides[key]:.4f} kcal/mol")

    unmatched = sorted(set(overrides) - matched)
    if unmatched:
        raise ValueError(
            f"adj_barrier name(s) {unmatched} not found in this reaction; "
            f"available short names {item_list}, full names {item_mol_name_list}"
        )
    return new_list


def write_thermo(
    PES_data,
    thermo_methods,
    thermo_path:Path,
    hindrot_item_reduced_mominert_dict,
    datfile:Path = Path("densum.dat"),
    verbose:bool = False,
    hrd_map:dict = None,
):
    """Write the main ``thermo`` input file describing the PES.

    Args:
        PES_data (dict): Potential energy surface data.
        thermo_methods (dict): Dictionary of thermo options.
        thermo_path (Path): Directory where files are written.
        hindrot_item_reduced_mominert_dict (dict): Reduced moments for hindered
            rotors.
        datfile (Path, optional): Output file name. Defaults to ``densum.dat``.
        verbose (bool, optional): Print progress messages. Defaults to ``False``.

    Returns:
        None
    """

    # Parse thermo_methods information
    if_tunneling = thermo_methods["tunneling"]
    if_hinderedrotor = thermo_methods["hinderedrotor"]
    if_anharm = thermo_methods["anharm"]
    adj_barrier = thermo_methods.get("adj_barrier", [])
    thermo_temp = thermo_methods["temperatures"]
    thermo_pressure = thermo_methods["pressures"]

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
    # item_Mol_list: 3 Molecules objects
    item_Mol_list = []
    # forward barrier list: 0.0, 1.78
    forwards_barrier_list = []
    for n, PES_num in enumerate(PES_data):
        for item in PES_data[PES_num]["PES_items"]:
            item_list.append(item)
            item_mol_name_list.append(PES_data[PES_num]["PES_items"][item]["mol_name"])
            Mol = PES_data[PES_num]["PES_items"][item]["Mol"]
            item_Mol_list.append(Mol)
            # A co-reactant written with "- X" (e.g. a bimolecular "TS - O2") shares the
            # reactant reference and sits at energy 0; the section's PES_energy already has
            # X subtracted, so it is the barrier of the "+" species (the TS) ALONE. Assigning
            # the section energy to the "- X" co-reactant too would double-count it and collapse
            # the barrier (e.g. O2 at +5.85 instead of 0 -> barrier 5.85-0-5.85 = 0).
            if PES_data[PES_num]["PES_items"][item].get("plus_minus") == "-":
                forwards_barrier = 0.0
            else:
                forwards_barrier = PES_data[PES_num]["PES_energy"]
            forwards_barrier_list.append(forwards_barrier)
        if PES_data[PES_num]["final_ts"]:
            reverse_PES_num = list(PES_data.keys())[n + 1]
            backwards_barrier = PES_data[reverse_PES_num]["reverse"]
            break

    # apply any empirical barrier adjustment to the named channel(s) (e.g.
    # 'Alkoxyl133a-HLeave_O2_another_ts=1.85') before the thermo fit -- this is
    # what changes the channel's k(T)/Ea (and the MORERXN Ea/R for the bimol O2
    # channel), with no hand-editing of the deck.
    forwards_barrier_list = apply_adj_barrier(
        item_list, item_mol_name_list, item_Mol_list, forwards_barrier_list,
        adj_barrier, verbose=verbose,
    )

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

    for n, (dummy_name, Mol, forwards_barrier) in enumerate(zip(item_list, item_Mol_list, forwards_barrier_list)):
        mol = Mol
        # [HRD] species match is by mol name (configparser lowercases keys)
        hrd_entry = (hrd_map or {}).get(str(item_mol_name_list[n]).lower())

        lines = get_thermo_lines(
            mol,
            dummy_name = dummy_name,
            thermo_path = thermo_path,
            forwards_barrier = forwards_barrier,
            backwards_barrier = backwards_barrier,
            if_tunneling = if_tunneling,
            if_anharm = if_anharm,
            if_hinderedrotor = if_hinderedrotor,
            hindrot_item_Mol_dict = hindrot_item_Mol_dict,
            hindrot_item_reduced_mominert_dict = hindrot_item_reduced_mominert_dict,
            hrd_modes = hrd_entry["modes"] if hrd_entry else None,
        )

        reaction_lines += lines

    reaction_lines.append(f"  {os.linesep}")

    # prepare thermo.dat
    with open(datfile, "w") as f:
        for line in reaction_lines:
            f.write(f"{line} {os.linesep}")
        f.write(f"  {os.linesep}")

def write_single_thermo(
    temp,
    mol:Molecules,
    dummy_name:str,
    thermo_path:Path = Path("thermo"),
    verbose:bool = False,
    hrd_modes:list = None,
):
    """Write a stand-alone ``thermo`` input file for ``mol``.

    Args:
        temp (str): Temperature grid string.
        mol (Molecules): Molecule to be processed.
        dummy_name (str): Identifier prefix.
        thermo_path (Path, optional): Output directory. Defaults to ``thermo``.
        verbose (bool, optional): Emit progress information. Defaults to
            ``False``.
        hrd_modes (list, optional): PES.in ``[HRD]``-declared rotors for this species.

    Returns:
        None
    """

    reaction_lines = get_thermo_head_lines(temp, 1)

    lines = get_thermo_lines(
        mol,
        dummy_name = dummy_name,
        thermo_path = thermo_path,
        mol_type = "none",
        hrd_modes = hrd_modes,
    )

    reaction_lines += lines

    reaction_lines.append(f"  {os.linesep}")

    # prepare thermo.dat
    data_name = f"{dummy_name}.therm"
    datfile = thermo_path.absolute() / data_name
    if verbose:
        print(f"{module:11} Write to: {datfile}")
    with open(datfile, "w") as f:
        for line in reaction_lines:
            f.write(f"{line} {os.linesep}")
        f.write(f"  {os.linesep}")
    return


def run_thermo(
    datfile:Path = Path("thermo.dat"),
    outfile:Path = None,
    verbose:bool = False,
):
    """Execute the ``thermo`` program using ``datfile`` as input.

    Args:
        datfile (Path): ``thermo`` input file.
        outfile (Path, optional): Where to move the output. Defaults to the
            same name with ``.out`` extension.
        verbose (bool, optional): Display executed command. Defaults to
            ``False``.

    Returns:
        None
    """

    cwd = datfile.parent.absolute()

    command = f"cd {cwd}; " + config.machine.thermo_command + f" {datfile.name}"
    if verbose:
        print(f"{module:11} Run command: {command}")
    subprocess.call(command, shell=True)

    if outfile is not None:
        # get default output file name
        _datname = str(datfile.absolute())
        default_outfile = Path(_datname[:len(_datname)-4] + ".out")

        # make sure the output file exists
        assert default_outfile.exists(), f"{default_outfile} doesn't exists!"

        # move the default output file to targeted outfile
        if verbose:
            print(f"{module:11} Write to {outfile}")
        default_outfile.rename(outfile.absolute())

def read_electronic_partition_function(
    outfile:Path = Path("thermo.out"),
    verbose:bool = False,
) -> float:
    """Read the temperature independent electronic partition function.

    Args:
        outfile (Path): Output file from ``thermo``.
        verbose (bool, optional): Print diagnostic information. Defaults to
            ``False``.

    Returns:
        float: The electronic partition function.
    """
    with open(outfile, "r") as f:
        lines = f.readlines()

    temp_list = []
    qele_list = []

    start_line_number = 0
    for n_line, line in enumerate(lines):
        if "Qelectr" in line:
            start_line_number = n_line + 1
            break

    for line in lines[start_line_number:]:
        data = line.split()
        if data != []:
            temp_list.append(float(data[0]))
            qele_list.append(float(data[8]))
        else:
            break

    # Qelectr is usually T-independent (its numbers are all equal), but it is
    # genuinely T-dependent for species with low-lying electronic states -- e.g.
    # OH (2-Pi spin-orbit, ~140 cm-1), O2, NO -- where q_elec rises with T. That
    # is physically correct and must NOT abort the workflow. KTOOLS never uses
    # this scalar (it writes the electronic ladder into its deck directly); only
    # the MULTIWELL deck writer consumes it. So warn and fall back to the low-T
    # value instead of asserting T-independence.
    qele = np.array(qele_list)
    if not np.allclose(qele, np.ones_like(qele) * qele[0]):
        print(
            f"{module:11} WARNING: electronic partition function is T-dependent "
            f"(low-lying electronic states): {qele}; using q_elec(T_min)={qele[0]:.4f}. "
            "KTOOLS ignores this scalar; MULTIWELL uses the low-T value."
        )

    return float(qele[0])

