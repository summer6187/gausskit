import pickle, re, json
import argparse
from pathlib import Path
import numpy as np

from gausskit.molecules import Molecules
from potential_energy_surface import match_method

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


def append_species(database, filepath:Path, method=None):
    raw_name = filepath.stem
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
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory where contains gaussian log files")
    args = parser.parse_args()

    database_path = Path(args.directory)
    print(f"Parsing every log files in this directory: {database_path.resolve()}")

    database = {}

    logfile_list = []
    for path_object in database_path.rglob('*'):
        if path_object.is_file() and path_object.suffix == ".log":
            logfile_list.append(path_object)


    for logfile in logfile_list:
        filepath = logfile.resolve()
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

