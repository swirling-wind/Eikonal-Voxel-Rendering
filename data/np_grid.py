import os
import numpy as np

def irrad_loc_dir_path(sampler_multiplier: int) -> str:
    grid_path = os.path.join(os.getcwd(), "data", "saves", f"NP(irrad,loc_dir)({sampler_multiplier}-samplers).npz")
    return grid_path

def save_irrad_loc_dir(irrad: np.ndarray, loc_dir: np.ndarray, sampler_multiplier: int):
    file_path = irrad_loc_dir_path(sampler_multiplier)
    np.savez_compressed(file_path, irrad=irrad, loc_dir=loc_dir)

def load_irrad_loc_dir(sampler_multiplier: int) -> tuple[np.ndarray, np.ndarray]:
    file_path = irrad_loc_dir_path(sampler_multiplier)
    npzfile = np.load(file_path)
    print("[ Loaded ] irradiance and local direction from", file_path.split("/")[-1])
    return npzfile['irrad'], npzfile['loc_dir']

def irrad_loc_dir_save_exists(sampler_multiplier: int) -> bool:
    file_path = irrad_loc_dir_path(sampler_multiplier)
    if os.path.exists(file_path):
        return True
    else:
        print("[ Not found ] irradiance and local direction. Start simulation...")
        return False