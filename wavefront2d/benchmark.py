import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
import time
import pickle

from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage

from wavefront2d.patch import *
from wavefront2d.point import *

def get_ground_truth(ground_truth_threshold, scene, ior_data, initial_wavefront_patches, num_steps, delta_t, field_size):
    ground_truth_filename = f"ground_truth_{scene}.npz"
    if os.path.exists(ground_truth_filename):
        data = np.load(ground_truth_filename)
        ground_truth_irradiance = data['irradiance']
        ground_truth_time = data['time']
        print(f"Loaded ground truth for {scene} from file.")
    else:
        print(f"Start generating ground truth for {scene}...")
        start_time = time.time()
        ground_truth_irradiance = accumulate_patches(ior_data['ior'], initial_wavefront_patches, num_steps, delta_t, refine_threshold=ground_truth_threshold, field_size=field_size)
        end_time = time.time()
        ground_truth_time = end_time - start_time
        
        np.savez(ground_truth_filename, irradiance=ground_truth_irradiance, time=ground_truth_time)
        print(f"Generated and saved new ground truth for {scene}.")
    
    return {
        'irradiance': ground_truth_irradiance,
        'time': ground_truth_time,
        'refine_threshold': ground_truth_threshold
    }

def benchmark_accumulate_patches(refine_thresholds, scene, ior_data, initial_wavefront_patches, num_steps, delta_t, field_size):
    results = []
    for threshold in refine_thresholds:
        print(f"Processing {scene} with refine_threshold={threshold}...")
        start_time = time.time()
        irradiance = accumulate_patches(ior_data['ior'], initial_wavefront_patches, num_steps, delta_t, refine_threshold=threshold, field_size=field_size)
        end_time = time.time()
        
        results.append({
            'refine_threshold': threshold,
            'time': end_time - start_time,
            'irradiance': irradiance
        })
    
    return results

def benchmark_accumulate_points_monte_carlo(num_iterations_list, scene, ior_data, initial_wavefront_positions, initial_wavefront_directions, num_steps, delta_t, field_size):
    results = []
    for num_iterations in num_iterations_list:
        print(f"Processing {scene} with num_iterations={num_iterations}...")
        start_time = time.time()
        irradiance = accumulate_points_monte_carlo(
            ior_data['ior'],
            initial_wavefront_positions,
            initial_wavefront_directions,
            num_steps,
            delta_t,
            num_simulations=num_iterations,
            perturbation_std=0.5,
            field_size=field_size
        )
        end_time = time.time()
        
        results.append({
            'num_iterations': num_iterations,
            'time': end_time - start_time,
            'irradiance': irradiance
        })
    
    return results
