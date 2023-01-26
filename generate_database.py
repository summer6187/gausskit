import os, pickle, re, sys
from gausskit.molecules import Molecules
from gausskit.PES_parser import match_method

# for filename in os.walk("."): # (dirpath, dirnames, filenames)
#     print(filename)

def simplify_raw_name(raw_name):
    match_string = ["g3xk", "g3x-k", "g4", "anharm", "opt"]
    for _string in match_string:
        if _string in raw_name.lower():
            start_i = re.search(_string, raw_name.lower()).start()
            raw_name = raw_name[:start_i]
        
        if raw_name[-1] == "_":
            raw_name = raw_name[:-1]
        if raw_name[-1] == "-":
            raw_name = raw_name[:-1]
    return raw_name

def parse_species(filepath):
    return Molecules.from_log(filepath)


def append_species(database, filepath, method=None):
    file_split_list = filepath.split("/")
    raw_name = file_split_list[-1].split(".")[-2]
    simple_name = simplify_raw_name(raw_name)
    new_mol = parse_species(filepath)
    method = new_mol.method

    if simple_name not in database:
        database[simple_name] = {}

    if method not in database[simple_name]:
        database[simple_name][method] = new_mol
    else:
        matched_method = match_method(database[simple_name], method)
        method_max = max(matched_method)
        method_n = method_max.split("_")[-1]
        if method_n.isdigit():
            _n = int(method_n) + 1
        else:
            _n = 1
        method = f"{method}_{str(_n)}"
        database[simple_name][method] = new_mol

    return database


if __name__ == "__main__":

    if len(sys.argv) < 2:
        database_path = "."    
    else:
        database_path = sys.argv[1]

    database = {}

    filenames_list = [
        os.path.join(dirpath, f)
        for (dirpath, dirnames, filenames) in os.walk(database_path)
        for f in filenames
    ]
    log_filename_list = [
        filename for filename in filenames_list if filename.split(".")[-1] == "log"
    ]

    for filepath in log_filename_list:
        print(f"Parsing filepath {filepath}")
        database = append_species(database, filepath)

    with open("database.pickle", "wb") as f:
        pickle.dump(database,f)