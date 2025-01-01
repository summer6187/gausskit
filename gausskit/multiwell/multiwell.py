from pathlib import Path
import subprocess

from gausskit.multiwell.workflow import fix_crp_file
from gausskit.multiwell.sctst import write_bdens, write_parsctst
from gausskit.settings import Configuration
from gausskit._defaults import colliders, trail_line

config = Configuration()

module = "[Multiwell]"

def write_multiwell(
    PES_data,
    multiwell_methods,
    multiwell_path:Path,
    hindrot_item_reduced_mominert_dict,
    collider="N2",
    Egrain="10   3000    4000    50000",
    datfile:Path = Path("densum.dat"),
    verbose:bool = False,
):
    # Parse multiwell_methods information
    multiwell_pressures = multiwell_methods["pressures"]
    multiwell_wells = multiwell_methods["wells"]
    multiwell_channels = multiwell_methods["channels"]
    multiwell_anharm = multiwell_methods["anharm"]

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
        if PES_data[PES_num]["final_ts"] == True:
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
        "298   298      !   <-  translational and initial vibrational temperatures.",
        "",
        f"{len(multiwell_pressures)}",  # number of pressure
        "  ".join(multiwell_pressures),  # pressure
        "",
        f"{len(multiwell_wells)}  {len(multiwell_channels)}", # here we assume there is one and only one product for each channel
    ]
    
    n_mol_dict = {}
    n_mol = 0 # number of wells or products in this section
    # formatting well line
    for n_well in multiwell_wells:
        n_mol += 1
        mol_list = item_Mol_list[n_well - 1]
        if type(mol_list)==list and len(mol_list) > 1:
            exit(f"This well is not unimolecule! Check it out! {mol_list}") # this is a shit code, rewrite it with expect

        dummy_name_keys = PES_data[str(n_well)]["PES_items"].keys()
        dummy_name = list(dummy_name_keys)[0]
        mol = PES_data[str(n_well)]["PES_items"][dummy_name]["Mol"]
        mol_name = PES_data[str(n_well)]["PES_items"][dummy_name]["mol_name"]
        
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

    reaction_lines.append("")

    # formatting collider model
    collider_line = colliders[collider]
    reaction_lines.append(collider_line)
    for n_well in multiwell_wells:
        # Mol: index number of Well
        n_well = n_well

        # Sig: Lennard-Jones $\sigma$ (Angstrom) for this well
        lj_sigma = "5.17"

        # Eps: Lennard-Jones $\epsilon$/kB (Kelvins) for this well
        lj_eps = "395.10"
        
        # ITYPE: selects model type in Subroutine PDOWN (see below for description of collision
        # models). Model types and explanations are given below.
        itype = 1

        line = f"{n_well}   {lj_sigma}   {lj_eps}   {itype}    "

        # DC(8): eight (8) coefficients for energy transfer model
        line += "100.  0.0   0.0   0.0   0.0   0.0   0.0   0.0"

        reaction_lines.append(line)

    reaction_lines.append(f"LJ")

    # formatting transition state lines
    reaction_lines.append("\n")
    reaction_lines.append(f"{len(multiwell_channels)}") # number of channels
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
        line += f"'rev' 'NOTUN' 'FAST' 'cent2' 'sum'"
        reaction_lines.append(line)

    # trial line
    reaction_lines.append("\n")
    reaction_lines.append(trail_line)

    multiwell_dat = multiwell_path / "multiwell.dat"
    if verbose:
        print(f"{module:11} Write to {multiwell_dat}")
    reaction_f = open(multiwell_dat, "w")
    reaction_f.write('\n'.join(reaction_lines) + '\n')
    reaction_f.close()


def run_multiwell(
    datfile:Path = Path("multiwell.dat"),
    verbose:bool = False,
):
    cwd = datfile.parent.absolute()

    command = f"cd {cwd}; " + config.machine.multiwell_command + f" {datfile.name}"
    if verbose:
        print(f"{module:10} Run command: {command}")
    subprocess.call(command, shell=True)
    return
