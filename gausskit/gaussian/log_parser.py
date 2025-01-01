import re
import argparse

from ase.io.gaussian import read_gaussian_out
from ase.units import Hartree


def check_normal_termination(lines):
    normal_term = False
    for line in lines:
        if "Normal termination" in line:
            normal_term = True

    if not normal_term:
        print('Didn\'t find "Normal termination"!')
    return normal_term


def get_result_blocks(f_lines):
    """
    takes log file lines
    find results summary blocks
    return results blocks list
    """
    block_begin_lines = []
    block_end_lines = []
    for i, line in enumerate(f_lines):
        if "1\\1\\" in line:
            block_begin_lines.append(i)
        if "@" in line:
            block_end_lines.append(i)

    result_blocks = []
    for i, j in zip(block_begin_lines, block_end_lines):
        result_block = ""
        for num in range(i, j + 1):
            result_block += f_lines[num]
        result_blocks.append(re.sub("\n ", "", result_block))
    return result_blocks


def get_energy(result_blocks, functional, basis_set, name):
    energy = 0
    result_list = []
    for i, result_block in enumerate(result_blocks):
        if functional in result_block and basis_set in result_block:
            result_list = re.split(r"\\", result_block)

    for result in result_list:
        if "=" in result and re.split("=", result)[0] == name:
            energy = float(re.split("=", result)[1])
    return energy


def get_thermal_data(lines):
    # get ZPE, DE_U, DE_H, DE_G
    for line in lines:
        if "Zero-point correction" in line:
            E_ZPE = float(re.findall(r"\d.*\d", line)[0])
        if "Thermal correction to Energy" in line:
            DE_U = float(re.findall(r"\d.*\d", line)[0])
        if "Thermal correction to Enthalpy" in line:
            DE_H = float(re.findall(r"\d.*\d", line)[0])
        if "Thermal correction to Gibbs Free Energy" in line:
            DE_G = float(re.findall(r"\d.*\d", line)[0])
            # break the loop after DE_G found in lines
            break
        if "Temperature" in line and "Pressure" in line:
            Temp = float(re.findall(r"\d\S*", line)[0])
            Pressure = float(re.findall(r"\d\S*", line)[1])

    thermal_data = {
        "E_ZPE": E_ZPE,
        "DE_U": DE_U,
        "DE_H": DE_H,
        "DE_G": DE_G,
        "Temp": Temp,
        "Pressure": Pressure,
    }

    return thermal_data


def get_anharm_ZPE(lines):
    for line in lines:
        if "Total Anharm   :" in line:
            # convert from cm-1 to hartree
            E_anharm_ZPE = float(re.findall(r"([0-9]+\.[0-9]+)", line)[0]) / 219474.6
            break

    return E_anharm_ZPE


def scale_ZPE(thermal_data, ZPE_scale):
    """
    Scale ZPE and correct other thermal data
    for composit methods like G4 or G3Xk
    """
    raw_ZPE = thermal_data["E_ZPE"]
    thermal_data["raw_ZPE"] = raw_ZPE
    thermal_data["E_ZPE"] = raw_ZPE * ZPE_scale
    thermal_data["DE_U"] += raw_ZPE * (ZPE_scale - 1)
    thermal_data["DE_H"] += raw_ZPE * (ZPE_scale - 1)
    thermal_data["DE_G"] += raw_ZPE * (ZPE_scale - 1)

    return thermal_data


def parse_freq(lines, ZPE_scale=None, anharm=False):
    # get thermal data
    thermal_data = get_thermal_data(lines)

    if ZPE_scale:
        thermal_data = scale_ZPE(thermal_data, ZPE_scale)

    # get anharmonic ZPE if used
    if anharm:
        ZPE_anharm = get_anharm_ZPE(lines)
        thermal_data["E_ZPE_anharm"] = ZPE_anharm

    return thermal_data


def verbose_print(Eele, thermal_data):
    print("%-25s %15.3f" % ("Temperature(K)=", thermal_data.Temp))
    print("%-25s %15.3f" % ("Pressure(atm)=", thermal_data.Pressure))
    print("%-25s %15.7f" % ("Eele=", Eele))
    print("%-25s %15.7f" % ("ZPE(scalced if Gn)=", thermal_data.E_ZPE))
    print("%-25s %15.7f" % ("Thermal correction U=", thermal_data.DE_U))
    print("%-25s %15.7f" % ("Thermal correction H=", thermal_data.DE_H))
    print("%-25s %15.7f" % ("Thermal correction G=", thermal_data.DE_G))
    print("%-25s %15.7f" % ("E0=", Eele + thermal_data.E_ZPE))
    print(
        "%-25s %15.7f"
        % ("U(" + str(thermal_data.Temp)[:6] + "K) =", Eele + thermal_data.DE_U)
    )
    print(
        "%-25s %15.7f"
        % ("H(" + str(thermal_data.Temp)[:6] + "K) =", Eele + thermal_data.DE_H)
    )
    print(
        "%-25s %15.7f"
        % ("G(" + str(thermal_data.Temp)[:6] + "K) =", Eele + thermal_data.DE_G)
    )


def get_E_SO():
    # get E_SO is set to 0
    return 0


def get_g3xk_ele_energy(result_blocks, E_SO):
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


def get_g4_energy(result_blocks):

    energy_dict = {
        "G4": get_energy(result_blocks, "G4", "", "G4"),
    }

    E0 = energy_dict["G4"]
    return E0


def read_g4_energy(filename, anharm=False, verbose=False):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

        # check if the calculation terminate normally
        # only check last lines is necessary
        check_normal_termination(lines[-10:])

        # get thermal data with scaled ZPE
        ZPE_scale = 0.9854
        thermal_data = parse_freq(lines, ZPE_scale, anharm)

        # read result blocks to calculate G4 Eele
        result_blocks = get_result_blocks(lines)
        E0 = get_g4_energy(result_blocks[-1:])

        Eele = E0 - thermal_data["E_ZPE"]

        # if verbose, print the result in a formated way
        if verbose:
            verbose_print(Eele, thermal_data)

        # generate energy summary
        energy = {
            "Eele": Eele,
            "E_ZPE": thermal_data["E_ZPE"],
            "E0": E0,
        }
        return energy


def check_g3xk_method(method_list):
    _g3xk = False
    g3xk_method_list = [
        "m062x/6-31g(2df,p)",
        "m062x/6-31g(2df,p)",
        "mp2(full)/gtlarge",
        "mp4(fc)/6-31g(2df,p)",
        "mp4(fc)/6-31+g(d)",
        "ccsd(t,e4t,maxcyc=999,t1diag)/6-31g(d)",
        "hf/gen",
    ]
    if len(g3xk_method_list) == len(method_list):
        _g3xk = True

        for g3xk_method, method in zip(g3xk_method_list, method_list):
            if g3xk_method not in method:
                _g3xk = False

    return _g3xk


def remove_duplicate_methods(method_list, basis_list):
    clean_mstring_list = []
    for method, basis in zip(method_list, basis_list):
        if method[0].lower() == "u" or method[0].lower() == "r":
            method = method[1:]
        mstring = "/".join([method, basis])
        if mstring not in clean_mstring_list:
            clean_mstring_list.append(mstring)

    new_m_list = []
    new_b_list = []
    for mstring in clean_mstring_list:
        new_m_list.append(mstring.split("/")[0])
        new_b_list.append(mstring.split("/")[1])

    return new_m_list, new_b_list

def read_charge_and_multiplicity(
    filename: str,
):
    """
    The read charge and multiplicity line in log file,
    for example:
        Charge =  0 Multiplicity = 1
    """

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if "Charge" in line:
            _line = line.split()
            if _line[3] == "Multiplicity":
                return int(_line[2]), int(_line[5])
            else:
                print(f"Charge and Multiplicity not found in this file {filename}")
                return None, None

def read_external_symmetry_number(
    filename: str,
    n_atoms,
) -> int:
    """
    The read external symmetry number line in log file,
    for example:
        Rotational symmetry number  1.
    """

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if "Rotational sy" in line or "ROTATIONAL SY" in line:
            _line = line.split()
            return int(float(_line[3]))

    if n_atoms <= 1:
        return 1

    print(f"External symmetry number (Rotational symmetry number) not found in this file {filename}")
    return 1

def read_log_energy(
    filename: str,
    method: str = None,
    freq: bool = False,
    anharm: bool = False,
    verbose: bool = False,
) -> dict:

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # check if the calculation terminate normally
    # only check last lines is necessary
    if not check_normal_termination(lines[-10:]):
        print(f"Error file: {filename}")
        return None

    # special method
    if method:
        if method.lower() == "g3xk" or method.lower() == "g3x-k":
            g3xk_energy = read_g3xk_energy(filename, anharm)
            Eele = g3xk_energy["Eele"]
            thermal_data = g3xk_energy

        elif method.lower() == "g4":
            g4_energy = read_g4_energy(filename, anharm)
            Eele = g4_energy["Eele"]
            thermal_data = g4_energy
        # normal calculation
        else:
            if freq:
                thermal_data = parse_freq(lines, anharm=anharm)
            # parse log file using ase.io.gaussian
            with open(filename, "r", encoding="utf-8") as f:
                gaussian_out = read_gaussian_out(f)
                Eele = gaussian_out.get_total_energy() / Hartree

    # if verbose, print the result in a formated way
    if verbose:
        verbose_print(Eele, thermal_data)

    # generate energy summary
    energy = {
        "Eele": Eele,
    }
    if freq:
        energy["ZPE"] = thermal_data["E_ZPE"]
        energy["E0"] = Eele + thermal_data["E_ZPE"]
        if anharm:
            energy["ZPE_anharm"] = thermal_data["E_ZPE_anharm"]
            energy["E0_anharm"] = Eele + thermal_data["E_ZPE_anharm"]

    return energy


# like shit but maybe it work
def read_log_parameters(
    filename,
    method: str = None,
    freq: bool = None,
    anharm: bool = None,
    hindrot: bool = None,
    verbose: bool = False,
):

    with open(filename) as f:
        lines = f.readlines()

    result_blocks = get_result_blocks(lines)

    parameters_list = []
    for result_block in result_blocks:
        parameters = {}
        parameter_line = result_block.split("\\\\")[1]
        if "anharm" in parameter_line.lower():
            parameters["anharm"] = True
        else:
            parameters["anharm"] = False

        if (
            "hindrot" in parameter_line.lower()
            or "hinderedrotor" in parameter_line.lower()
        ):
            parameters["hindrot"] = True
        else:
            parameters["hindrot"] = False

        if "freq" in parameter_line.lower():
            parameters["freq"] = True
        else:
            parameters["freq"] = False

        for item in parameter_line.split():
            found_method = False
            if item.lower() == "g4":
                found_method = True
                parameters["method"] = "G4"
                parameters["basis"] = ""
                break
            elif "/" in item:
                found_method = True
                parameters["method"] = item.split("/")[0]
                parameters["basis"] = item.split("/")[-1]
        if not found_method:
            parameters["method"] = result_block.split("\\")[4]
            parameters["basis"] = result_block.split("\\")[5]

        parameters_list.append(parameters)

    overall_parameters = {}
    _anharm = False
    _hindrot = False
    _freq = False
    _g4 = False
    _method = ""
    _basis = ""
    _method_list = []
    overall_method = []
    overall_basis = []
    for parameters in parameters_list:
        if parameters["anharm"]:
            _anharm = True
        if parameters["hindrot"]:
            _hindrot = True
        if parameters["freq"]:
            _freq = True
        if parameters["method"] == "G4":
            _g4 = True
        _method_list.append(
            "/".join([parameters["method"], parameters["basis"]]).lower()
        )
        overall_method.append(parameters["method"])
        overall_basis.append(parameters["basis"])

    _g3xk = check_g3xk_method(_method_list)

    if _g4:
        _method = "G4"
        _basis = ""
    elif _g3xk:
        _method = "G3X-K"
        _basis = ""
    else:
        overall_method, overall_basis = remove_duplicate_methods(
            overall_method, overall_basis
        )
        _method = ",".join(overall_method)
        _basis = ",".join(overall_basis)

    if anharm == None:
        overall_parameters["anharm"] = _anharm
    else:
        overall_parameters["anharm"] = anharm

    if method == None:
        overall_parameters["method"] = _method
        overall_parameters["basis"] = _basis
    else:
        overall_parameters["method"] = method
        overall_parameters["basis"] = ""

    if freq == None:
        overall_parameters["freq"] = _freq
    else:
        overall_parameters["freq"] = freq

    if hindrot == None:
        overall_parameters["hindrot"] = _hindrot
    else:
        overall_parameters["hindrot"] = hindrot

    return overall_parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="example.log gaussian output file")
    args = parser.parse_args()

    filename = args.file
    parameters = read_log_parameters(filename)
    anharm = parameters["anharm"]
    freq = parameters["freq"]
    method = parameters["method"]
    basis = parameters["basis"]
    hindrot = parameters["hindrot"]

    print(parameters)
    energy = read_log_energy(
        filename,
        method=method,
        freq=freq,
        anharm=anharm,
    )
    print(energy)
