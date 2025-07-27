"""Utility functions to extract energies from G3XK Gaussian outputs."""

import sys
from gausskit.gaussian.log_parser import get_energy, check_normal_termination, parse_freq, get_result_blocks, verbose_print


def get_E_SO():
    """Return the spin--orbit correction.

    Returns:
        float: Always ``0`` because spin--orbit contributions are ignored.
    """

    return 0


def get_g3xk_ele_energy(result_blocks, E_SO):
    """Compute the electronic energy using the G3XK scheme.

    Args:
        result_blocks (list[str]): Parsed ``\1\`` blocks from the log file.
        E_SO (float): Spin--orbit correction energy.

    Returns:
        float: The final electronic energy in Hartree.
    """
    scale_factor_dict = {
        "SHF": 1.0945,
        "SE234": 1.0712,
        "SCC": 1.2347,
        "SE2": 1.1467,
        "SE3": 1.2249,
        "SE4": 0.4135,
    }

    energy_dict = {
        "HF/6-31G(d)": get_energy(result_blocks, "CCSD(T)-FC", "6-31G(d)", "HF"),
        "MP2/6-31G(d)": get_energy(result_blocks, "CCSD(T)-FC", "6-31G(d)", "MP2"),
        "MP3/6-31G(d)": get_energy(result_blocks, "CCSD(T)-FC", "6-31G(d)", "MP3"),
        "MP4/6-31G(d)": get_energy(result_blocks, "CCSD(T)-FC", "6-31G(d)", "MP4SDTQ"),
        "CCSD(T)/6-31G(d)": get_energy(
            result_blocks, "CCSD(T)-FC", "6-31G(d)", "CCSD(T)"
        ),
        "HF/G3large": get_energy(result_blocks, "MP2-Full", "GTlarge", "HF"),
        "MP2(full)/G3Large": get_energy(result_blocks, "MP2-Full", "GTlarge", "MP2"),
        "MP2/6-31+G(d)": get_energy(result_blocks, "MP4SDTQ-FC", "6-31+G(d)", "MP2"),
        "MP3/6-31+G(d)": get_energy(result_blocks, "MP4SDTQ-FC", "6-31+G(d)", "MP3"),
        "MP4/6-31+G(d)": get_energy(
            result_blocks, "MP4SDTQ-FC", "6-31+G(d)", "MP4SDTQ"
        ),
        "MP2/6-31G(2df,p)": get_energy(
            result_blocks, "MP4SDTQ-FC", "6-31G(2df,p)", "MP2"
        ),
        "MP3/6-31G(2df,p)": get_energy(
            result_blocks, "MP4SDTQ-FC", "6-31G(2df,p)", "MP3"
        ),
        "MP4/6-31G(2df,p)": get_energy(
            result_blocks, "MP4SDTQ-FC", "6-31G(2df,p)", "MP4SDTQ"
        ),
        "HF/G3XL": get_energy(result_blocks, "HF", "Gen", "HF"),
    }

    terms = {
        "E2/d": energy_dict["MP2/6-31G(d)"] - energy_dict["HF/6-31G(d)"],
        "E3/d": energy_dict["MP3/6-31G(d)"] - energy_dict["MP2/6-31G(d)"],
        "E4/d": energy_dict["MP4/6-31G(d)"] - energy_dict["MP3/6-31G(d)"],
        "ECC/d": energy_dict["CCSD(T)/6-31G(d)"] - energy_dict["MP4/6-31G(d)"],
        "E2(FU)/G3L": energy_dict["MP2(full)/G3Large"] - energy_dict["HF/G3large"],
        "E3/+": energy_dict["MP3/6-31+G(d)"] - energy_dict["MP2/6-31+G(d)"],
        "E3/2dfp": energy_dict["MP3/6-31G(2df,p)"] - energy_dict["MP2/6-31G(2df,p)"],
        "E4/+": energy_dict["MP4/6-31+G(d)"] - energy_dict["MP3/6-31+G(d)"],
        "E4/2dfp": energy_dict["MP4/6-31G(2df,p)"] - energy_dict["MP3/6-31G(2df,p)"],
    }

    Eele = (
        energy_dict["HF/6-31G(d)"]
        + scale_factor_dict["SHF"]
        * (energy_dict["HF/G3XL"] - energy_dict["HF/6-31G(d)"])
        + scale_factor_dict["SE234"] * (terms["E2/d"] + terms["E3/d"] + terms["E4/d"])
        + scale_factor_dict["SCC"] * (terms["ECC/d"])
        + scale_factor_dict["SE2"] * (terms["E2(FU)/G3L"] - terms["E2/d"])
        + scale_factor_dict["SE3"] * (terms["E3/+"] - terms["E3/d"])
        + scale_factor_dict["SE3"] * (terms["E3/2dfp"] - terms["E3/d"])
        + scale_factor_dict["SE4"] * (terms["E4/+"] - terms["E4/d"])
        + scale_factor_dict["SE4"] * (terms["E4/2dfp"] - terms["E4/d"])
        + E_SO
    )

    return Eele


def read_g3xk_energy(filename, anharm=False, verbose=False):
    """Parse a G3XK output file and calculate energies.

    Args:
        filename (str | Path): Path to the Gaussian log file.
        anharm (bool, optional): Whether anharmonic corrections were
            included in the run. Defaults to ``False``.
        verbose (bool, optional): Print a formatted summary of the
            energies. Defaults to ``False``.

    Returns:
        dict: Dictionary with keys ``Eele``, ``E_ZPE`` and ``E0``.
    """

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

        # check if the calculation terminate normally
        # only check last lines is necessary
        check_normal_termination(lines[-10:])

        # get thermal data
        # scale ZPE and correct other thermal data
        ZPE_scale = 0.972
        thermal_data = parse_freq(lines, ZPE_scale, anharm)

        # get E_SO
        E_SO = get_E_SO()

        # read result blocks to calculate G3XK Eele
        result_blocks = get_result_blocks(lines)
        Eele = get_g3xk_ele_energy(result_blocks, E_SO)

        # if verbose, print the result in a formated way
        if verbose:
            verbose_print(Eele, thermal_data)

        # generate energy summary
        energy = {
            "Eele": Eele,
            "E_ZPE": thermal_data["E_ZPE"],
            "E0": Eele + thermal_data["E_ZPE"],
        }
        return energy


if __name__ == "__main__":
    filename = sys.argv[1]
    read_g3xk_energy(filename, verbose=False)
