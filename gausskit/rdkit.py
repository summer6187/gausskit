# identify optical isomers using rdkit

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from thermo.group_contribution.joback import Joback

# import gausskit.molecules as molecules

def atoms2rdkmol(
    mol,
    charge:int = 0,
):
    """
    Script from xyz2mol
    cite: https://github.com/jensengroup/xyz2mol
    """
    xyz_block = mol.get_xyz_block()

    raw_rdkit_mol = Chem.MolFromXYZBlock(xyz_block)
    rdkit_mol = Chem.Mol(raw_rdkit_mol)

    charge = Chem.GetFormalCharge(rdkit_mol)
    try:
        rdDetermineBonds.DetermineBonds(rdkit_mol, charge=charge)
    except ValueError as err:
        # print(f"Mol {mol} with charge {charge} show error {err}")
        trail_charge = int(str(err).split()[8][1:-2])
        try:
            rdDetermineBonds.DetermineBonds(rdkit_mol, charge=trail_charge)
        except ValueError as err:
            print(f"Mol {mol} with charge {charge} show error {err}")

    return rdkit_mol

def get_chiral_centers(
    mol,
    charge:int = 0,
):
    rdkit_mol = atoms2rdkmol(mol, charge)

    chiral_centers = Chem.FindMolChiralCenters(rdkit_mol, includeUnassigned=True)
    return chiral_centers

def get_optical_isomers(
    mol,
):
    chiral_centers = get_chiral_centers(mol)

    if chiral_centers == []:
        return 1
    else:
        return 2

def get_Joback(
    mol,
    charge = 0,
):
    """
    Estimates the normal boiling temperature of an organic compound using 
    the Joback method as a function of chemical structure only.
    cite: https://thermo.readthedocs.io/thermo.group_contribution.joback.html
    """
    rdkit_mol = atoms2rdkmol(mol, charge)
    J = Joback(rdkit_mol)
    return J.estimate()

def get_lj_parameters(mol):
    J = get_Joback(mol)

    # Cite: https://github.com/ReactionMechanismGenerator/RMG-Py/blob/bb35064105e1b79c59312446df704ad05fed6754/rmgpy/data/transport.py#L427
    Tc = J["Tc"]
    Pc = J["Pc"] * 1.0e-5
    # epsilon = .77 * Tc
    sigma = 2.44 * (Tc / Pc) ** (1. / 3)

    # Vc = J["Vc"] * 1e6
    # sigma = 1.45 * (Vc ** (1/3))
    Tb = J["Tb"]
    epsilon = 1.21 * Tb

    return sigma, epsilon
