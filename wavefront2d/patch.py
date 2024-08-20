import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
from .initialize import compute_gradients

class Patch:
    def __init__(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                 start_dir: np.ndarray, end_dir: np.ndarray, energy: float):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.start_dir = start_dir
        self.end_dir = end_dir
        self.energy = energy
    
    def mid_pos(self):
        return (self.start_pos + self.end_pos) / 2
    
    def length(self):
        return np.linalg.norm(self.end_pos - self.start_pos)
    
    def __repr__(self):
        return (f"Patch - start_pos{self.start_pos.tolist()}, "
                f"end_pos{self.end_pos.tolist()} | "
                f"start_dir{self.start_dir.tolist()}, "
                f"end_dir{self.end_dir.tolist()} | "
                f"mid_pos{self.mid_pos().tolist()} | "
                f"energy={self.energy}, "
                f"length={self.length()}")
    
def init_wavefront_patches(initial_wavefront_positions: list[tuple], initial_wavefront_directions: list[tuple],
                            field_size=128) -> list[Patch]:
    initial_wavefront_patches = []
    for i in range(field_size - 1):
        start_pos = np.array(initial_wavefront_positions[i])
        end_pos = np.array(initial_wavefront_positions[i + 1])
        start_dir = np.array(initial_wavefront_directions[i])
        end_dir = np.array(initial_wavefront_directions[i + 1])
        energy = 1.0  # Initial energy of each patch
        patch = Patch(start_pos, end_pos, start_dir, end_dir, energy)
        initial_wavefront_patches.append(patch)
    return initial_wavefront_patches

def update_wavefront_patches(patches: list[Patch], IOR: np.ndarray, IOR_grad: tuple[np.ndarray, np.ndarray], 
                             delta_t: float, refine_threshold: float = 0.2, energy_threshold: float = 0.0005) -> list[Patch]:
    new_patches = []
    grad_x_ior, grad_y_ior = IOR_grad
    
    for patch in patches:
        # Split patches until all patches are smaller than the refine_threshold
        patch_queue = [patch]
        while patch_queue:
            current_patch = patch_queue.pop(0)
            start_x, start_y = current_patch.start_pos
            end_x, end_y = current_patch.end_pos
            start_vx, start_vy = current_patch.start_dir
            end_vx, end_vy = current_patch.end_dir

            if current_patch.energy < energy_threshold:
                continue  # Eliminate the patch and continue with the next one

            if np.linalg.norm([end_x - start_x, end_y - start_y]) > refine_threshold:
                # If the patch is too long, split it into two smaller patches
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                mid_vx = (start_vx + end_vx) / 2
                mid_vy = (start_vy + end_vy) / 2

                new_patch_1 = Patch(np.array([start_x, start_y]), np.array([mid_x, mid_y]),
                                    np.array([start_vx, start_vy]), np.array([mid_vx, mid_vy]),
                                    current_patch.energy / 2)
                new_patch_2 = Patch(np.array([mid_x, mid_y]), np.array([end_x, end_y]),
                                    np.array([mid_vx, mid_vy]), np.array([end_vx, end_vy]),
                                    current_patch.energy / 2)
                patch_queue.append(new_patch_1)
                patch_queue.append(new_patch_2)
            else:
                if (0 <= int(start_y) < IOR.shape[0] and 0 <= int(start_x) < IOR.shape[1] and
                    0 <= int(end_y) < IOR.shape[0] and 0 <= int(end_x) < IOR.shape[1]):

                    start_n = IOR[int(start_y), int(start_x)]
                    end_n = IOR[int(end_y), int(end_x)]
                    start_nx = grad_x_ior[int(start_y), int(start_x)]
                    start_ny = grad_y_ior[int(start_y), int(start_x)]
                    end_nx = grad_x_ior[int(end_y), int(end_x)]
                    end_ny = grad_y_ior[int(end_y), int(end_x)]

                    # x_i+1 = x_i + delta_t * v_i / n^2
                    # Calculate the new positions
                    new_start_x = start_x + delta_t * (start_vx / (start_n**2))
                    new_start_y = start_y + delta_t * (start_vy / (start_n**2))
                    new_end_x = end_x + delta_t * (end_vx / (end_n**2))
                    new_end_y = end_y + delta_t * (end_vy / (end_n**2))
                    # v_i+1 = v_i + delta_t * grad_n / n
                    # Calculate the new directions
                    new_start_vx = start_vx + delta_t * (start_nx / start_n)
                    new_start_vy = start_vy + delta_t * (start_ny / start_n)
                    new_end_vx = end_vx + delta_t * (end_nx / end_n)
                    new_end_vy = end_vy + delta_t * (end_ny / end_n)

                    new_patch = Patch(np.array([new_start_x, new_start_y]), np.array([new_end_x, new_end_y]),
                                      np.array([new_start_vx, new_start_vy]), np.array([new_end_vx, new_end_vy]),
                                      current_patch.energy)
                    new_patches.append(new_patch)

    return new_patches

def simulate_wavefront_propagation_patches(cur_IOR: np.ndarray, initial_wavefront_patches: list[Patch],
                                           num_steps, delta_t) -> list[list[Patch]]:
    wavefront_patch_list = [initial_wavefront_patches]
    cur_IOR_grad = compute_gradients(cur_IOR)
    for _ in range(num_steps):
        new_patches = update_wavefront_patches(wavefront_patch_list[-1], cur_IOR, cur_IOR_grad, delta_t)
        wavefront_patch_list.append(new_patches)
    return wavefront_patch_list

def compute_irradiance_patches(wavefront_patch_list: list[list[Patch]], field_size: int) -> np.ndarray:
    irradiance = np.zeros((field_size, field_size))
    for patch_list in wavefront_patch_list:
        for patch in patch_list:
            mid_x, mid_y = patch.mid_pos()            
            # Compute the voxel coordinates of the patch midpoint
            voxel_x, voxel_y = int(mid_x), int(mid_y)            
            # Check if the voxel coordinates are within the field boundaries
            if 0 <= voxel_x < field_size and 0 <= voxel_y < field_size:
                # Update the irradiance for the voxel containing the patch midpoint
                irradiance[voxel_y, voxel_x] += patch.energy    
    return irradiance

def accumulate_patches(cur_IOR: np.ndarray, initial_wavefront_patches: list[Patch],
                       num_steps, delta_t, field_size=128) -> np.ndarray:

    cur_IOR_grad = compute_gradients(cur_IOR)

    prev_patch_list = initial_wavefront_patches
    irradiance = np.zeros((field_size, field_size))
    for _ in range(num_steps):
        new_patches = update_wavefront_patches(prev_patch_list, cur_IOR, cur_IOR_grad, delta_t)
        for patch in new_patches:
            mid_x, mid_y = patch.mid_pos()
            
            # Compute the voxel coordinates of the patch midpoint
            voxel_x, voxel_y = int(mid_x), int(mid_y)
            
            # Check if the voxel coordinates are within the field boundaries
            if 0 <= voxel_x < field_size and 0 <= voxel_y < field_size:
                # Update the irradiance for the voxel containing the patch midpoint
                irradiance[voxel_y, voxel_x] += patch.energy   

        prev_patch_list = new_patches 
    return irradiance
