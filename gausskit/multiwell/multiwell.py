from pathlib import Path
import subprocess

from gausskit.multiwell.thermo import write_thermo, run_thermo, parse_adj_barrier
from gausskit.rdkit import get_lj_parameters
from gausskit.settings import Configuration
from gausskit._defaults import colliders

config = Configuration()

module = "[Multiwell]"

def write_multiwell(
    PES_data,
    multiwell_methods,
    collider="N2",
    Egrain="10   3000    4000    50000",
    datfile:Path = Path("multiwell.dat"),
    bimolecular_concentration:float = 1.,
    verbose:bool = False,
):
    """Generate the ``multiwell`` input file for a reaction network.

    Args:
        PES_data (dict): Potential energy surface information.
        multiwell_methods (dict): Options controlling the calculation.
        collider (str, optional): Name of the collider gas. Defaults to ``"N2"``.
        Egrain (str, optional): Energy grain specification. Defaults to
            ``"10   3000    4000    50000"``.
        datfile (Path, optional): Destination file name. Defaults to
            ``multiwell.dat``.
        bimolecular_concentration (float, optional): Concentration factor for
            bimolecular channels. Defaults to ``1.``.
        verbose (bool, optional): Emit progress information. Defaults to
            ``False``.

    Returns:
        None
    """

    # Parse multiwell_methods information
    multiwell_temperature = multiwell_methods["temperature"]
    multiwell_pressures = multiwell_methods["pressures"]
    multiwell_wells = multiwell_methods["wells"]
    multiwell_channels = multiwell_methods["channels"]
    multiwell_tunneling = multiwell_methods["tunneling"]
    multiwell_anharm = multiwell_methods["anharm"]
    # adj_barrier: {NAME: barrier} overrides for the forward (unimolecular) channels.
    # The named species may be any channel's TS (matched by short or full name); the
    # bimolecular O2 channel's barrier is handled separately via the bimol thermo. Names
    # not matching a forward channel here are simply skipped (they may belong to the bimol
    # reaction), so we do NOT validate-and-raise in this path.
    adj_overrides = parse_adj_barrier(multiwell_methods.get("adj_barrier", []))

    # gather PES_info
    # item_list: Mol1, TS2, Mol3
    item_list = []
    # item_mol_name_list: HCFC133a, HCFC133a-OH_ts, Radical133a
    item_mol_name_list = []
    # item_Mol_list: 3 Molecules objects
    item_Mol_list = []
    # barrier list: include forward and backward barrier
    barrier_list = []
    for n, PES_num in enumerate(PES_data):
        for item in PES_data[PES_num]["PES_items"]:
            item_list.append(item)
            item_mol_name_list.append(PES_data[PES_num]["PES_items"][item]["mol_name"])
            Mol = PES_data[PES_num]["PES_items"][item]["Mol"]
            item_Mol_list.append(Mol)
        if PES_data[PES_num]["final_ts"] and n + 1 < len(PES_data):
            forwards_barrier = PES_data[PES_num]["PES_energy"]
            reverse_PES_num = list(PES_data.keys())[n + 1]
            backwards_barrier = PES_data[reverse_PES_num]["reverse"]
        else:
            # for well or product mol, set dummy barrier == 0
            forwards_barrier = 0
            backwards_barrier = 0
        barrier_list.append([forwards_barrier, backwards_barrier])

    reaction_lines = [
        "Gausskit generated. Be careful.",
        f"{Egrain}     1832960486",  # a random seed
        "",
        "'ATM '  'KCAL'  'AMUA'",
        "",
        f"{multiwell_temperature}    {multiwell_temperature}   !   <-  translational and initial vibrational temperatures.",
        "",
        f"{len(multiwell_pressures)}",  # number of pressure
        "  ".join(multiwell_pressures),  # pressure
        "",
    ]

    num_product = len(multiwell_channels)

    if multiwell_methods.get("bimolecular_channel"):
        print(f"{module:10} Warning!!! Currently only one bimolecular reation implementated")
        num_product += 1

    reaction_lines.append(f"{len(multiwell_wells)}  {num_product}") # here we assume there is one and only one product for each channel
    
    n_mol_dict = {}
    n_mol = 0 # number of wells or products in this section
    # formatting well line
    for n_well in multiwell_wells:
        n_mol += 1
        mol_list = item_Mol_list[n_well - 1]
        if isinstance(mol_list, list) and len(mol_list) > 1:
            exit(f"This well is not unimolecule! Check it out! {mol_list}") # this is a shit code, rewrite it with expect

        dummy_name_keys = PES_data[str(n_well)]["PES_items"].keys()
        dummy_name = list(dummy_name_keys)[0]
        mol = PES_data[str(n_well)]["PES_items"][dummy_name]["Mol"]
        PES_data[str(n_well)]["PES_items"][dummy_name]["mol_name"]
        
        # HMol: enthalpy of formation at 0 K (units defined by keyword)
        relative_energy = 0.0

        # MolMom: rotational parameter for 2-dimensional external rotation 
        # (moment of inertia or rotational constant; units defined by 
        # keyword on Line 3)
        rotational_parameter = mol.ad_rotor

        # Molsym: external symmetry number for well
        external_symmetry_number = mol.external_symmetry_number

        # Molele: electronic partition function for well(REAL number); depends on temperature;
        # can be obtained from THERMO output.
        electronic_partition_function = mol.electronic_partition_function

        # Molopt: number of chiral stereoisomers (or "optical isomers", or mirror images)
        # for well
        chiral_stereoisomers = mol.optical_isomers

        line = f"{n_mol}  {dummy_name:>10}  {relative_energy:.4f}  {rotational_parameter:.4f} \
        {external_symmetry_number}   {electronic_partition_function}   {chiral_stereoisomers}"
        reaction_lines.append(line)

        n_mol_dict[n_well] = n_mol
    
    # formatting product lines
    for (n_well, n_ts, n_product) in multiwell_channels:
        dummy_name_keys = PES_data[str(n_product)]["PES_items"].keys()
        product_dummy_name = "+".join(list(dummy_name_keys))

        # mol = PES_data[str(n_product)]["PES_items"][dummy_name]["Mol"]
        # mol_name = PES_data[str(n_product)]["PES_items"][dummy_name]["mol_name"]

        n_mol += 1
        # Hmol: enthalpy of formation at 0 K (units defined by keyword on Line 3)
        # [ignored unless tunneling is used]
        relative_energy = PES_data[str(n_product)]["PES_energy"]
        line = f"{n_mol}  {product_dummy_name:>10}  {relative_energy:.4f}"
        reaction_lines.append(line)

        n_mol_dict[n_product] = n_mol

    if multiwell_methods.get("bimolecular_channel"):
        bimolecular_channel = multiwell_methods["bimolecular_channel"]

        # here we fix the last item in the channel to be the product
        n_product = bimolecular_channel[-1]

        dummy_name_keys = PES_data[str(n_product)]["PES_items"].keys()
        product_dummy_name = "+".join(list(dummy_name_keys))

        # mol = PES_data[str(n_product)]["PES_items"][dummy_name]["Mol"]
        # mol_name = PES_data[str(n_product)]["PES_items"][dummy_name]["mol_name"]

        n_mol += 1
        # Hmol: enthalpy of formation at 0 K (units defined by keyword on Line 3)
        # [ignored unless tunneling is used]
        relative_energy = PES_data[str(n_product)]["PES_energy"]
        line = f"{n_mol}  {product_dummy_name:>10}  {relative_energy:.4f}"
        reaction_lines.append(line)

        n_mol_dict[n_product] = n_mol

    reaction_lines.append("")

    # formatting collider model
    collider_line = colliders[collider]

    # get reactant mass
    n_well = multiwell_wells[0]
    dummy_name_keys = PES_data[str(n_well)]["PES_items"].keys()
    dummy_name = list(dummy_name_keys)[0]
    mol = PES_data[str(n_well)]["PES_items"][dummy_name]["Mol"]
    collider_line += f"{mol.get_masses().sum():.4f}     ! {collider} Collider"

    reaction_lines.append(collider_line)
    for n_well in multiwell_wells:
        # Mol: index number of Well
        n_well = n_well

        dummy_name_keys = PES_data[str(n_well)]["PES_items"].keys()
        dummy_name = list(dummy_name_keys)[0]
        mol = PES_data[str(n_well)]["PES_items"][dummy_name]["Mol"]

        # Sig: Lennard-Jones $\sigma$ (Angstrom) for this well
        # Eps: Lennard-Jones $\epsilon$/kB (Kelvins) for this well
        lj_sigma, lj_eps = get_lj_parameters(mol)

        # ITYPE: selects model type in Subroutine PDOWN (see below for description of collision
        # models). Model types and explanations are given below.
        itype = 1

        line = f"{n_well}   {lj_sigma:.4f}   {lj_eps:.4f}   {itype}    "

        # DC(8): eight (8) coefficients for energy transfer model
        line += "100.  0.0   0.0   0.0   0.0   0.0   0.0   0.0"

        reaction_lines.append(line)

    reaction_lines.append("LJ")

    # formatting transition state lines
    reaction_lines.append("\n")
    # Number of FORWARD reaction channels declared for the wells (the transition-state
    # channels only). Any bimolecular reaction is emitted below as a MORERXN supplementary
    # reaction, NOT a forward channel, so it must NOT be counted here.
    n_declared_channels = len(multiwell_channels)
    reaction_lines.append(f"{n_declared_channels}") # number of forward (TS) channels
    for (n_well, n_ts, n_product) in multiwell_channels:
        dummy_name_keys = PES_data[str(n_ts)]["PES_items"].keys()
        dummy_name = list(dummy_name_keys)[0]

        mol = PES_data[str(n_ts)]["PES_items"][dummy_name]["Mol"]

        # RR: 2-D external rotational parameter
        rotational_parameter = mol.ad_rotor

        # j: external symmetry number for TS
        external_symmetry_number = mol.external_symmetry_number

        # Qel: electronic partition function for TS (REAL number)
        electronic_partition_function = mol.electronic_partition_function

        # l: number of optical isomers for TS
        chiral_stereoisomers = mol.optical_isomers

        # AA: A-factor for reaction (units: s-1); only used for ILT method, but ALWAYS read in.
        # For SCTST/CRP channels MultiWell expects a large placeholder here (matches the
        # reference anharmonic decks); for harmonic 'sum' channels it is unused, so 0.0.
        a_factor = 1.0e+16 if multiwell_anharm else 0.0

        # EE: reaction critical energy (E0), relative to ZPE of reactant (Mol) (i.e. the barrier
        # height with zpe corrections). When using CRP files generated by program
        # SCTST, E0 is set to the larger of zero, or the enthalpy difference (at 0 K) between
        # product and reactant; i.e. E0 = MAX[ 0.0 , (∆H(ito) – ∆H((Mol)].
        if multiwell_anharm:
            # CRP/SCTST channel: the ZPE-corrected barrier is carried by the .crp (Vf/Vr written
            # into parsctst.dat), so the deck E0 is the 0 K reaction enthalpy, not the barrier
            # height: E0 = MAX[0, dH] (product - reactant well). MultiWell manual Section 8.3.
            e_well = PES_data[str(n_well)]["PES_energy"]
            e_product = PES_data[str(n_product)]["PES_energy"]
            relative_critical_energy = max(0.0, e_product - e_well)
        else:
            # harmonic 'sum' channel: E0 is the ZPE-corrected barrier height itself.
            relative_critical_energy = PES_data[str(n_ts)]["PES_energy"]
        # adj_barrier override for this forward channel (named by short or full TS name).
        ts_mol_name = PES_data[str(n_ts)]["PES_items"][dummy_name].get("mol_name", dummy_name)
        if dummy_name in adj_overrides:
            relative_critical_energy = adj_overrides[dummy_name]
        elif ts_mol_name in adj_overrides:
            relative_critical_energy = adj_overrides[ts_mol_name]

        line = f"{n_mol_dict[n_well]}  {n_mol_dict[n_product]}  {dummy_name:>10}  {rotational_parameter:.4f}   "
        line += f"{external_symmetry_number}   {electronic_partition_function}   {chiral_stereoisomers}   "
        line += f"{a_factor}   {relative_critical_energy:.4f}   "

        Habs_tunneling = False
        if multiwell_tunneling:
            # check if there is any H abstraction reaction
            dummy_name_keys = PES_data[str(n_product)]["PES_items"].keys()
            for dummy_name in list(dummy_name_keys):
                prod_mol = PES_data[str(n_product)]["PES_items"][dummy_name]["Mol"]
                if prod_mol.get_chemical_formula() == "H":
                    Habs_tunneling = True

        # centrifugal keyword: NOCENT for the H-abstraction TS, cent2 otherwise (same in
        # harmonic and anharmonic). The DOS keyword and tunnelling differ by method:
        cent = "'NOCENT'" if Habs_tunneling else "'cent2'"
        if multiwell_anharm:
            # SCTST: the cumulative reaction probability (read from <TS>.crp) already
            # contains reaction-coordinate tunnelling, so the channel MUST be 'NOTUN' with
            # no separate 'TUN' line, and the DOS keyword is 'CRP' instead of 'sum'.
            # (Emitting 'sum'/'TUN' here -- the old behaviour -- makes MultiWell-2023.1 read
            # the harmonic sum-of-states and FATAL "TUN CANNOT BE USED WITH CRP".)
            line += f"'rev' 'NOTUN' 'FAST' {cent} 'CRP'"
            reaction_lines.append(line)
        elif Habs_tunneling:
            line += f"'rev' 'TUN' 'FAST' {cent} 'sum'"
            reaction_lines.append(line)
            freq = mol.frequencies
            img_freq = freq[freq<0][0]
            assert img_freq < 0, f"Frequency problem: {freq}"
            reaction_lines.append(f"'TUN'    {-img_freq:.4f}")
        else:
            line += f"'rev' 'NOTUN' 'FAST' {cent} 'sum'"
            reaction_lines.append(line)


    # bimolecular competing reaction if required -- emitted as a MORERXN
    # supplementary reaction (the working mechanism in MULTIWELL 2023.1).
    #
    # A supplementary 2nd-order reaction (norder=2) is the bath-gas / co-reactant
    # pseudo-first-order channel: the stepper computes ksup = Afc*exp(-Bsr/T) and
    # uses rsup = numbdens*ksup (bookstep_mod.f), i.e. rate = Afc*[bath]*exp(-Bsr/T).
    # To make the rate depend on the co-reactant (e.g. O2) concentration rather
    # than the total bath, Afc carries the 2nd-order rate constant scaled by the
    # co-reactant mole fraction (bimolecular_concentration); numbdens then supplies
    # the total number density, recovering k2*[O2].
    #
    # Format after the forward-channel block:
    #     'MORERXN'
    #     <n_supplementary>
    #     Mol  product  'Name'  order  Afc  Bsr(=Ea/R, K)  quench
    #
    # THERMO fits k(T)=A*exp(B/T); read_rate_from_thermo returns (A, -B), so the
    # second value IS Ea/R in Kelvin -- exactly Bsr. (The earlier negative-A 'ILT'
    # forward-channel encoding produced a NaN k(E) in 2023.1, where the matching
    # `Press = -AA` bath factor is computed but never applied to the rate.)
    if multiwell_methods.get("bimolecular_channel"):
        bimolecular_channel = multiwell_methods["bimolecular_channel"]

        # here we fix the last item in the channel to be the product
        n_product = bimolecular_channel[-1]
        n_well = bimolecular_channel[0]

        A, B = multiwell_methods["bimolecular_rates"]   # A (cm3 s-1), B = Ea/R (K)

        # scale the 2nd-order A-factor by the co-reactant concentration (mole fraction)
        A *= bimolecular_concentration

        bimol_dummy_name = "bimol"

        reaction_lines.append("'MORERXN'")
        reaction_lines.append("1")  # number of supplementary reactions
        reaction_lines.append(
            f"{n_mol_dict[n_well]}  {n_mol_dict[n_product]}  '{bimol_dummy_name}'  2  "
            f"{A:.6e}  {B:.4f}  0.0"
        )

    # trial line
    reaction_lines.append("")
    reaction_lines.append(multiwell_methods["trails"])

    if verbose:
        print(f"{module:11} Write to {datfile}")
    reaction_f = open(datfile, "w")
    reaction_f.write('\n'.join(reaction_lines) + '\n')
    reaction_f.close()


def run_multiwell(
    datfile:Path = Path("multiwell.dat"),
    verbose:bool = False,
):
    """Call the ``multiwell`` executable on ``datfile``.

    Args:
        datfile (Path): Input file for ``multiwell``.
        verbose (bool, optional): Print the command being executed. Defaults to
            ``False``.

    Returns:
        None
    """

    cwd = datfile.parent.absolute()

    command = f"cd {cwd}; " + config.machine.multiwell_command + f" {datfile.name}"
    if verbose:
        print(f"{module:10} Run command: {command}")
    subprocess.call(command, shell=True)
    return


def get_thermo_methods(
    multiwell_methods: dict,
    verbose:bool = False,
):
    """Translate multiwell settings into a dictionary for :mod:`thermo`.

    Args:
        multiwell_methods (dict): Options read from the MultiWell section.
        verbose (bool, optional): Show the translated settings. Defaults to
            ``False``.

    Returns:
        dict: Thermo configuration derived from MultiWell settings.
    """
    thermo_methods = {
        "tunneling": False,
        "hinderedrotor": False,
        "anharm": multiwell_methods["anharm"],
        "adj_barrier": multiwell_methods.get("adj_barrier", []),
        "temperatures": f"{multiwell_methods['temperature']}",
        "pressures": multiwell_methods["pressures"],
    }
    if verbose:
        print(f"{module:10} Thermo methods from multiwell methods")
        print(f"{module:10} {thermo_methods =}")
    return thermo_methods


def read_rate_from_thermo(
    outfile:Path,
):
    """Extract Arrhenius parameters from a ``thermo`` output file.

    Args:
        outfile (Path): File written by the ``thermo`` program.

    Returns:
        tuple: ``(A, B)`` Arrhenius parameters.
    """
    with open(outfile, "r") as f:
        lines = f.readlines()

    for nn, line in enumerate(lines):
        if "A(T)" in line:
            dataline = lines[nn+1]
            break

    data_list = dataline.split()
    A, B = data_list[2], data_list[3]
    return float(A), -float(B)


def run_bimol_thermo(
    PES_data:dict,
    multiwell_methods:dict,
    thermo_path:Path,
    verbose:bool = False,
):
    """Compute rates for bimolecular channels using ``thermo``.

    Args:
        PES_data (dict): Potential energy surface information.
        multiwell_methods (dict): Input options for MultiWell.
        thermo_path (Path): Working directory for ``thermo`` files.
        verbose (bool, optional): Enable log messages. Defaults to ``False``.

    Returns:
        None
    """
    bimol_PES_data = {}
    for PES_num in multiwell_methods["bimolecular_channel"]:
        bimol_PES_data[PES_num] = PES_data[str(PES_num)]

    # prepare thermo input file reaction.dat
    datfile = "bimol_reaction.dat"
    write_thermo(
        bimol_PES_data,
        thermo_methods=get_thermo_methods(multiwell_methods),
        thermo_path=thermo_path,
        hindrot_item_reduced_mominert_dict={},
        datfile=thermo_path / datfile,
        verbose=verbose,
    )

    run_thermo(
        datfile=thermo_path / datfile,
        verbose=verbose,
    )

    outfile = thermo_path / Path(datfile).with_suffix(".out")
    A, B = read_rate_from_thermo(
        outfile,
    )

    multiwell_methods["bimolecular_rates"] = (A, B)
