import os
import numpy as np

def irrad_loc_dir_path(sampler_multiplier: int) -> str:
    grid_path = os.path.join(os.getcwd(), "data", "saves", f"NP(irrad,loc_dir)({sampler_multiplier}-samplers).npz")
    return grid_path

def save_irrad_loc_dir(irrad: np.ndarray, loc_dir: np.ndarray, sampler_multiplier: int):
    np.savez_compressed(irrad_loc_dir_path(sampler_multiplier), irrad=irrad, loc_dir=loc_dir)

def load_irrad_loc_dir(sampler_multiplier: int) -> tuple[np.ndarray, np.ndarray]:
    npzfile = np.load(irrad_loc_dir_path(sampler_multiplier))
    print("[ Loaded ] irradiance and local direction from", irrad_loc_dir_path(sampler_multiplier).split("/")[-1])
    return npzfile['irrad'], npzfile['loc_dir']

def irrad_loc_dir_save_exists(sampler_multiplier: int) -> bool:
    if os.path.exists(irrad_loc_dir_path(sampler_multiplier)):
        return True
    else:
        print("[ Not found ] irradiance and local direction. Start simulation...")
        return False