import os
import numpy as np

def get_irrad_loc_dir_path(scene_config: dict) -> str:
    file_name = f"NP({scene_config['Name']})({scene_config['Sampler Num']}-samplers).npz"
    grid_path = os.path.join(os.getcwd(), "data", "saves", file_name)
    return grid_path

def save_irrad_loc_dir(irrad: np.ndarray, loc_dir: np.ndarray, file_path: str):
    np.savez_compressed(file_path, irrad=irrad, loc_dir=loc_dir)

def load_irrad_loc_dir(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    npzfile = np.load(file_path)
    print("[ Loaded ] irradiance and local direction from", file_path)
    return npzfile['irrad'], npzfile['loc_dir']

def irrad_loc_dir_save_exists(file_path: str) -> bool:
    if os.path.exists(file_path):
        return True
    else:
        print("[ Not found ] irradiance and local direction. Start simulation...")
        return False