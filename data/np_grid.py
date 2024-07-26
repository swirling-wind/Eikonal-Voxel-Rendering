import os
import numpy as np

IRRAD_LOC_DIR = "NP(irrad,loc_dir)"

def get_path(variable_name: str) -> str:
    directory_path = os.getcwd() + "\\data\\"
    filename = os.path.join(directory_path, f"{variable_name}")
    return filename

def save_irrad_loc_dir(irrad: np.ndarray, loc_dir: np.ndarray):
    np.savez_compressed(get_path(IRRAD_LOC_DIR), irrad=irrad, loc_dir=loc_dir)

def load_irrad_loc_dir(filename: str) -> tuple[np.ndarray, np.ndarray]:
    npzfile = np.load(get_path(filename) + ".npz")
    return npzfile['irrad'], npzfile['loc_dir']

def delete_data_file(filename: str):
    print("Try to remove file:", get_path(filename) + ".npz")
    if file_exists(filename):
        os.remove(get_path(filename) + ".npz")
    else:
        print("File does not exist")

def file_exists(filename: str) -> bool:
    return os.path.exists(get_path(filename) + ".npz")