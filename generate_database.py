import os, pickle, re, sys, json
from gausskit.molecules import Molecules
from PES_parser import match_method
import numpy as np

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
    if list(new_mol.hinderedrotor()) == []:
        pass
    else:
        method += "_hindrot"

    if simple_name not in database:
        database[simple_name] = {}

    if method not in database[simple_name]:
        database[simple_name][method] = new_mol
    else:
        database[simple_name][method] = new_mol
        

    return database

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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

    # write database in pickle binary file
    with open("database.pickle", "wb") as f:
        pickle.dump(database,f)

    # database_dict = {}
    # for item in database:
    #     database_dict[item] = {}
    #     for method in database[item]:
    #         database_dict[item][method] = database[item][method].to_dict()
    # dumped = json.dumps(database_dict,indent=4)
    # with open("database.json", "w") as f:
    #     f.write(dumped)

