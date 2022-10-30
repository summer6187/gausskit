import pickle
import sys

def get_name_info(ds: dict, name: str):
    name_info_list = []
    for _method in ds[name].keys():
        method_info = _method
        if ds[name][_method].ts:
            method_info += "_ts"
        if ds[name][_method].frequencies.any():
            method_info += "_freq"
        name_info_list.append(method_info)

    return name_info_list

if __name__ == "__main__":
    filename = sys.argv[1]
    with open(filename, "rb") as f:
        ds = pickle.load(f)


    for name in ds.keys():
        name_info = get_name_info(ds, name)
        print(f"{name:30}: {name_info}")