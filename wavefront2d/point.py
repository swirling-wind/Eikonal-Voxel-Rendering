import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
from .initialize import compute_gradients

def update_wavefront_points(pos: list[tuple], dir: list[tuple], IOR: np.ndarray, IOR_grad: tuple[np.ndarray, np.ndarray], delta_t: float) -> tuple[list[tuple], list[tuple]]:
    new_pos = []
    new_dir = []
    grad_x_ior, grad_y_ior = IOR_grad
    for (x, y), (vx, vy) in zip(pos, dir):
        if 0 <= int(y) < IOR.shape[0] and 0 <= int(x) < IOR.shape[1]:
            n = IOR[int(y), int(x)]
            
            # x_i+1 = x_i + delta_t * v_i / n^2
            # v_i+1 = v_i + delta_t * grad_n / n
            # calculate the new position
            new_x = x + delta_t * (vx / (n**2))
            new_y = y + delta_t * (vy / (n**2))

            # calculate the new direction
            nx = grad_x_ior[int(y), int(x)]
            ny = grad_y_ior[int(y), int(x)]
            new_vx = vx + delta_t * (nx / n)
            new_vy = vy + delta_t * (ny / n)

            new_pos.append((new_x, new_y))
            new_dir.append((new_vx, new_vy))

        else:
            pass

    return new_pos, new_dir

def simulate_wavefront_propagation_points(cur_IOR: np.ndarray, inital_wavefront_pos: list[tuple], initial_wavefront_dir: list[tuple],
                                          num_steps, delta_t) -> tuple[list[list[tuple]], list[list[tuple]]]:
    wavefront_pos_list = [inital_wavefront_pos]
    wavefront_dir_list = [initial_wavefront_dir]
    cur_IOR_grad = compute_gradients(cur_IOR)
    for _ in range(num_steps):
        wavefront_positions, wavefront_directions = update_wavefront_points(wavefront_pos_list[-1], wavefront_dir_list[-1], cur_IOR, cur_IOR_grad, delta_t)
        wavefront_pos_list.append(wavefront_positions)
        wavefront_dir_list.append(wavefront_directions)
    return wavefront_pos_list, wavefront_dir_list

def compute_irradiance_points(wavefront_pos_list: list[list[tuple]], field_size: int, irradiance_grid_size: int|None = None) -> np.ndarray:
    irradiance = np.zeros((field_size, field_size))
    for pos_list in wavefront_pos_list:
        for x, y in pos_list:
            if 0 <= int(y) < field_size and 0 <= int(x) < field_size:
                irradiance[int(y), int(x)] += 1
    return irradiance

def accumulate_points(cur_IOR: np.ndarray, inital_wavefront_pos: list[tuple], initial_wavefront_dir: list[tuple],
                      num_steps, delta_t, field_size=128) -> np.ndarray:
    irradiance = np.zeros((field_size, field_size))

    prev_wavefront_pos = inital_wavefront_pos
    prev_wavefront_dir = initial_wavefront_dir
    cur_IOR_grad = compute_gradients(cur_IOR)
    for _ in range(num_steps):
        wavefront_positions, wavefront_directions = update_wavefront_points(prev_wavefront_pos, prev_wavefront_dir, cur_IOR, cur_IOR_grad, delta_t)
           
        for x, y in prev_wavefront_pos:
            if 0 <= int(y) < field_size and 0 <= int(x) < field_size:
                irradiance[int(y), int(x)] += 1

        prev_wavefront_pos = wavefront_positions
        prev_wavefront_dir = wavefront_directions
    return irradiance