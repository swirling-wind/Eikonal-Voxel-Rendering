import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage


def get_ior_field(field_type: str, field_size = 128) -> np.ndarray:
    height, width = field_size, field_size

    center = (height//2, width//2)
    radius = int(field_size / 3) 
    y, x = np.ogrid[:height, :width]
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    
    if field_type == "linear":        
        gradient = np.linspace(0, 0.3, width, dtype=np.float32)
        linear_ior_field = np.tile(gradient, (height, 1)) + 1
        return linear_ior_field
    elif field_type == "board":
        tile_size = int(field_size / 2)
        board_ior_field =  np.kron([[1, 0] * (width//tile_size//2),
                        [0, 1] * (width//tile_size//2)] * (height//tile_size//2),
                        np.ones((tile_size, tile_size))) * 0.3 + 1.0
        return board_ior_field
    elif field_type == "convex":
        convex_lens_ior_field = np.ones((height, width), dtype=np.float32)
        convex_lens_ior_field[mask] = 1.5
        return convex_lens_ior_field
    elif field_type == "concave":
        concave_lens_ior_field = np.ones((height, width), dtype=np.float32) * 1.5
        concave_lens_ior_field[mask] = 1.0
        return concave_lens_ior_field
    elif field_type == "slope":        
        slope_ior_field = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                if y < x - int(field_size / 4):
                    slope_ior_field[y, x] = 1.5
                else:
                    slope_ior_field[y, x] = 1.0
        return slope_ior_field
    elif field_type == "steep_slope":
        steep_slope_ior_field = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                if y < 2 * x - int(field_size * 0.75):
                    steep_slope_ior_field[y, x] = 1.3
                else:
                    steep_slope_ior_field[y, x] = 1.0
        return steep_slope_ior_field

    else:
        raise ValueError("Invalid field type")
    

def compute_gradients(IOR: np.ndarray) -> tuple[np.ndarray, np.ndarray]:    
    smoothed_ior = gaussian_filter(IOR, sigma=4.0)
    grad_x = np.gradient(smoothed_ior, axis=1)
    grad_y = np.gradient(smoothed_ior, axis=0)
    return grad_x, grad_y
    

