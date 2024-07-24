import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage


def generate_initial_wavefront(num_samplers_per_voxel : int, pos_perturbation_scale : float, num_x=128, num_y=128, num_z=128) -> tuple[np.ndarray, np.ndarray]:
    sampler_multiplier = num_samplers_per_voxel
    position_perturbation = np.random.uniform(-pos_perturbation_scale, pos_perturbation_scale, (num_x * num_z * (sampler_multiplier**3), 3))
    initial_wavefront_pos = np.array([(x / sampler_multiplier, y / sampler_multiplier + num_y - 1.5, z / sampler_multiplier) 
                                      for x in range(num_x * sampler_multiplier) 
                                      for y in range(sampler_multiplier) 
                                      for z in range(num_z * sampler_multiplier)]) + position_perturbation
    initial_wavefront_dir = np.array([(0, -1, 0) for _ in range(num_x * num_z * (sampler_multiplier**3))])
    return initial_wavefront_pos, initial_wavefront_dir