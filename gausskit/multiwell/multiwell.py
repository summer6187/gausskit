from pathlib import Path
import subprocess

from gausskit.multiwell.thermo import write_thermo, run_thermo
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
    multiwell_methods["anharm"]

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
    # Number of reaction channels declared for the wells. MULTIWELL sizes its
    # per-well channel arrays (e.g. kuni) from this declared count; the count
    # MUST equal the TOTAL number of forward reactions originating from the
    # wells -- the transition-state channels PLUS any bimolecular reaction that
    # is added below. If the bimolecular reaction is omitted from this count,
    # MULTIWELL allocates kuni with too few channels and the 2023.1 solver
    # overflows ('Index N of dimension 2 of array kuni above upper bound').
    n_declared_channels = len(multiwell_channels)
    if multiwell_methods.get("bimolecular_channel"):
        # one bimolecular reaction (currently only one is supported)
        n_declared_channels += 1
    reaction_lines.append(f"{n_declared_channels}") # number of channels (TS + bimolecular)
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

        # AA: A-factor for reaction (units: s-1); only used for ILT method, but ALWAYS read in
        a_factor = 0.0

        # EE: reaction critical energy (E0), relative to ZPE of reactant (Mol) (i.e. the barrier
        # height with zpe corrections). When using CRP files generated by program
        # SCTST, E0 is set to the larger of zero, or the enthalpy difference (at 0 K) between
        # product and reactant; i.e. E0 = MAX[ 0.0 , (∆H(ito) – ∆H((Mol)].
        relative_critical_energy = PES_data[str(n_ts)]["PES_energy"]

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

        if Habs_tunneling:
            line += "'rev' 'TUN' 'FAST' 'NOCENT' 'sum'"
            reaction_lines.append(line)
            freq = mol.frequencies
            img_freq = freq[freq<0][0]
            assert img_freq < 0, f"Frequency problem: {freq}"
            reaction_lines.append(f"'TUN'    {-img_freq:.4f}")
        else:
            line += "'rev' 'NOTUN' 'FAST' 'cent2' 'sum'"
            reaction_lines.append(line)


    # bimolecular competing reaction if required.
    #
    # The bimolecular reaction is emitted as an additional FORWARD reaction
    # channel from the well (not as a separate MORERXN supplementary block).
    # This is required for correctness with the MULTIWELL 2023.1 solver: a
    # MORERXN reaction increments the well's supplementary-reaction count
    # (nsmax) but is NOT included in the per-well channel allocation
    # (largest = max(Nchan)). The stepper then indexes kuni up to
    # Nchan(Mol) + nsmax(Mol), overrunning the array and aborting with
    # 'Index 4 of dimension 2 of array kuni above upper bound of 3'.
    # By declaring the bimolecular reaction as a forward channel it is counted
    # in Nchan (and in the declared channel count above), so kuni is allocated
    # large enough and the solver runs.
    if multiwell_methods.get("bimolecular_channel"):
        bimolecular_channel = multiwell_methods["bimolecular_channel"]

        # here we fix the last item in the channel to be the product
        n_product = bimolecular_channel[-1]
        n_well = bimolecular_channel[0]

        A, B = multiwell_methods["bimolecular_rates"]

        # concentration of bath gas
        A *= bimolecular_concentration

        bimol_dummy_name = "bimol" #FIXME this need to update to valid dummy name

        # Emit the bimolecular reaction as a forward ILT channel using a
        # NEGATIVE A-factor, which is MULTIWELL's microcanonical pseudo-first-
        # order mechanism for a reaction of the well with the bath gas [B]:
        # a negative AA sets Press = -AA (the 2nd-order rate constant, cm3 s-1)
        # and the stepper multiplies the channel rate by the bath number
        # density, recovering the pseudo-first-order rate A*[B]. This keeps the
        # reaction counted in the well's Nchan (so kuni is allocated large
        # enough; see note at the channel-count line) while preserving the
        # canonical (A, B) computed by THERMO for the bimolecular reaction.
        # Columns: Mol  ito  Name  RR  sym  Qel  l  AA  EE  <keywords>.
        # AA carries the (negated) 2nd-order A-factor; EE is the activation
        # energy E0 in kcal/mol. THERMO fits k(T) = A*exp(B/T) and
        # read_rate_from_thermo returns B = -B(T), so Ea = R*B (R in kcal/mol/K).
        R_kcal = 1.987204e-3  # gas constant, kcal/mol/K
        activation_energy = R_kcal * B  # kcal/mol
        line = f"{n_mol_dict[n_well]}   {n_mol_dict[n_product]}    {bimol_dummy_name:>10}   1.0   "
        line += f"1   1.0   1   "
        line += f"{-A:.6e}   {activation_energy:.4f}   "
        line += "'NOREV' 'NOTUN' 'FAST' 'NOCENT' 'ILT'"

        reaction_lines.append(line)

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
        "adj_barrier": [],
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
