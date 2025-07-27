import numpy as np
import re

_re_freq = re.compile(r"-?[0-9]*\.[0-9]")

def parse_anharm_matrix_lines(anharm_matrix_lines:list):
    anharm_matrix_ll = []
    for line in anharm_matrix_lines:
        line_split = line.split()
        if line_split != []:
            anharm_matrix_ll.append(line_split)

    anharm_matrix_ll_array = []
    index_line_n_list = []
    max_index = 0
    for n, line in enumerate(anharm_matrix_ll):
        new_line = []
        all_digit = True
        for item in line:
            if item.isdigit():
                new_line.append(int(item))
            else:
                new_line.append(float(item))
                all_digit = False
        if all_digit:
            index_line_n_list.append(n)
            max_index = new_line[-1]

        anharm_matrix_ll_array.append(new_line)

    anharm_matrix = np.zeros((max_index, max_index))

    for n, index_line_n in enumerate(index_line_n_list):
        if n != len(index_line_n_list) - 1:
            next_index_line_n = index_line_n_list[n + 1]
            for line in anharm_matrix_ll_array[index_line_n + 1 : next_index_line_n]:
                i = line[0]
                for _j, item in enumerate(line[1:]):
                    j = anharm_matrix_ll_array[index_line_n][_j]
                    anharm_matrix[i - 1, j - 1] = item
                    # symmetrize anharmonic matrix, modify upper triangular matrix
                    anharm_matrix[j - 1, i - 1] = item
        else:
            for line in anharm_matrix_ll_array[index_line_n + 1 :]:
                i = line[0]
                for _j, item in enumerate(line[1:]):
                    j = anharm_matrix_ll_array[index_line_n][_j]
                    anharm_matrix[i - 1, j - 1] = item
                    # symmetrize anharmonic matrix, modify upper triangular matrix
                    anharm_matrix[j - 1, i - 1] = item

    return anharm_matrix

def read_anharm_x_matrix(filename):
    """
    This function takes a gaussian log file with anharmonic analysis
    filename: example.log
    this function return a lower triangular matrix numpy.array
    return anharm_matrix: numpy.array (in cm-1)
    """
    with open(filename) as f:
        found_anharm_matrix = False
        append_bool = False
        anharm_matrix_lines = []
        lines = f.readlines()
        for line in lines:
            if "Total Anharmonic X Matrix" in line:
                found_anharm_matrix = True
                append_bool = True
            if found_anharm_matrix:
                if "============================================" in line:
                    append_bool = False
                    break

            if append_bool:
                line = re.sub(r"D", "E", line)
                anharm_matrix_lines.append(line)
    anharm_matrix_lines = anharm_matrix_lines[2:]
    return parse_anharm_matrix_lines(anharm_matrix_lines)

def read_anharm_xl_matrix(filename):
    """
    This function read anharmonic Xl matrix where Fermi resonance exists
    """
    with open(filename) as f:
        found_anharm_matrix = False
        append_bool = False
        anharm_matrix_lines = []
        lines = f.readlines()
        for line in lines:
            if "Total Anharmonic Xl Matrix" in line:
                found_anharm_matrix = True
                append_bool = True
            if found_anharm_matrix:
                if "============================================" in line:
                    append_bool = False
                    break

            if append_bool:
                line = re.sub(r"D", "E", line)
                anharm_matrix_lines.append(line)
    anharm_matrix_lines = anharm_matrix_lines[2:]
    return parse_anharm_matrix_lines(anharm_matrix_lines)

def read_full_anharm_matrix(filename):
    """
    This function reads Fundamental Bands information,
    and recover full anharmonic matrix from
    Anharmonic X matrix and Anharmonic Xl matrix
    """
    anharm_matrix = read_anharm_x_matrix(filename)
    read_anharm_xl_matrix(filename)
    # anharm_matrix += anharm_xl_matrix

    with open(filename) as f:
        found_harm_freq = False
        append_bool = False
        harm_freq_lines = []
        lines = f.readlines()
        for line in lines:
            if "Fundamental Bands" in line:
                found_harm_freq = True
            if found_harm_freq:
                append_bool = True
                if "Overtones" in line:
                    append_bool = False
                    break

            if append_bool:
                harm_freq_lines.append(line)

    f_index_list = []
    for line in harm_freq_lines[3:-1]:
        line_split = line.split()
        i = line_split.index("active") - 1

        try:
            f_index = int(line_split[i].split("(")[0])
        except ValueError as err:
            exit(f"Parsing Fundamental Bands {filename} shows {err}")

        f_index_list.append(f_index)

    full_anharm_matrix = np.zeros((len(f_index_list), len(f_index_list)))

    used_i_list = []
    for ni, i in enumerate(f_index_list):
        used_j_list = []
        for nj, j in enumerate(f_index_list):
            if i in used_i_list and j in used_j_list:
                full_anharm_matrix[ni,nj] = anharm_matrix[i-1,j-1]
            else:
                full_anharm_matrix[ni,nj] = anharm_matrix[i-1,j-1]
            used_j_list.append(j)
        used_i_list.append(i)
    return full_anharm_matrix

def read_anharm_matrix(filename):
    """
    This function takes a gaussian log file with anharmonic analysis
    filename: example.log
    this function return a full anharmonic matrix numpy.array
    return anharm_matrix: numpy.array (in cm-1)
    """
    with open(filename) as f:
        found_anharm_x_matrix = False
        found_anharm_xl_matrix = False
        lines = f.readlines()
        for line in lines:
            if "Total Anharmonic X Matrix" in line:
                found_anharm_x_matrix = True
            if "Total Anharmonic Xl Matrix" in line:
                found_anharm_xl_matrix = True
                break

    if found_anharm_x_matrix:
        if found_anharm_xl_matrix:
            anharm_matrix = read_full_anharm_matrix(filename)
        else:
            anharm_matrix = read_anharm_x_matrix(filename)
        return anharm_matrix
    else:
        print(f"Anharmonic X Matrix not found in {filename}!")


def format_anharm_matrix(anharm_matrix):
    """
    This function takes a symmetric matrix generated by "read_anharm_matrix"
    anharm_matrix: numpy.array
    return lower triangular matrix list like formated output like (in cm-1):
    1.11111E+1
    1.11111E+1 1.11111E+1
    1.11111E+1 1.11111E+1 1.11111E+1
    """
    _anharm_matrix = np.tril(anharm_matrix)
    format_output = []
    for a in _anharm_matrix:
        format_line = a[a != 0]
        format_line = list(format_line)
        format_output.append(" ".join([f"{i:.5E}" for i in format_line]))
    return format_output


def read_harm_freq(filename):
    """
    This function reads a gaussian log file with anharmonic analysis
    return harmonic frequencies (cm-1)
    """
    with open(filename) as f:
        found_harm_freq = False
        append_bool = False
        harm_freq_list = []
        lines = f.readlines()
        for line in lines:
            if "Fundamental Bands" in line:
                found_harm_freq = True
            if found_harm_freq:
                append_bool = True
                if "Overtones" in line:
                    append_bool = False
                    break

            if append_bool:
                harm_freq_list.append(line)

    harm_freq_list_clean = []
    for line in harm_freq_list[3:-1]:
        line_split = line.split()
        i = line_split.index("active") + 1

        # in some cases there will be "3722.837**********"
        if "*" in line_split[i]:
            _clean_freq_list = re.findall(_re_freq, line_split[i])
            if _clean_freq_list != []:
                _clean_freq = _clean_freq_list[0]
                line_split[i] = _clean_freq
            else:
                print("Frequency parser failed!")
                print(line_split[i])

        try:
            freq = float(line_split[i])
        except ValueError:
            freq = float(line[30:38])

        harm_freq_list_clean.append(freq)

    return harm_freq_list_clean


def read_harm_freq_another(filename):
    with open(filename) as f:
        found_harm_freq = False
        append_bool = False
        harm_freq_list = []
        lines = f.readlines()
        for line in lines:
            if "Harmonic frequencies" in line: # TODO: read normal coordinates
                found_harm_freq = True
                append_bool = True
            if found_harm_freq:
                if "-------------------" in line:
                    append_bool = False
                    break

            if append_bool:
                harm_freq_list.append(line)

    harm_freq_list_clean = []
    for line in harm_freq_list[3:-1]:
        if "Frequencies" in line:
            _freq_list = re.findall(_re_freq, line)
            if _freq_list != []:
                for _freq in _freq_list:
                    harm_freq_list_clean.append(float(_freq))

    return harm_freq_list_clean


def read_method(filename):
    with open(filename) as f:
        lines = f.readlines()
    i_list = []
    j_list = []
    for n, line in enumerate(lines):
        if "1\\1\\" in line:
            i_list.append(n)
        if "@" in line:
            j_list.append(n)

    result_sum = []
    for i, j in zip(i_list, j_list):
        sub_lines = []
        for line in lines[i : j + 1]:
            line = re.sub(r"\n", "", line)
            line = re.sub(r"\ ", "", line)
            sub_lines.append(line)
        result_sum.append("".join(sub_lines))

    # ts_bool = False
    for result_i in result_sum:
        result_list = result_i.split("\\")
        result_list_lower = [item.lower() for item in result_list]
        if "freq" in result_list_lower:
            i = result_list_lower.index("freq")
            method = result_list[i + 1]
            basissets = result_list[i + 2]
        # if "fts" in result_list_lower:
        #     ts_bool = True
    return method, basissets  # , ts_bool


def format_freq_matrix(harm_freq, anharm_matrix):
    formated_lines = []
    for freq in harm_freq:
        formated_lines.append(f"{freq}")
    formated_lines.append("lower")
    formated_lines = formated_lines + format_anharm_matrix(anharm_matrix)
    return formated_lines


