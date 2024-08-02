import numpy as np

def compute_ior_gradient(ior_field: np.ndarray) -> np.ndarray:
    grad_xyz = np.gradient(ior_field)
    return np.stack(grad_xyz, axis=-1)

def remove_under_floor(grid: np.ndarray, floor_height: int) -> np.ndarray:
    grid[:, :floor_height, :] = 0
    return grid

def normalize_by_max(array: np.ndarray, max: int=255) -> np.ndarray:
    array = array.astype(float)    
    min_val = np.min(array)
    max_val = np.max(array)
    assert max_val > min_val, "max_val should be greater than min_val"
    normalized = (array - min_val) / (max_val - min_val)    # normalize to 0-1
    return np.round(normalized * max).astype(float)  # normalize to 0-max
