# identify optical isomers using rdkit

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from thermo.group_contribution.joback import Joback

# import gausskit.molecules as molecules

def atoms2rdkmol(mol):
    """
    Script from xyz2mol
    cite: https://github.com/jensengroup/xyz2mol
    """
    xyz_block = mol.get_xyz_block()

    raw_rdkit_mol = Chem.MolFromXYZBlock(xyz_block)
    rdkit_mol = Chem.Mol(raw_rdkit_mol)

    return rdkit_mol

def get_chiral_centers(
    mol,
    charge:int = 0,
):
    rdkit_mol = atoms2rdkmol(mol)

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

def get_Tb(mol):
    """
    Estimates the normal boiling temperature of an organic compound using 
    the Joback method as a function of chemical structure only.
    cite: https://thermo.readthedocs.io/thermo.group_contribution.joback.html
    """
    rdkit_mol = atoms2rdkmol(mol)
    J = Joback(rdkit_mol)
    return J.estimate(callables=False)["Tb"]
