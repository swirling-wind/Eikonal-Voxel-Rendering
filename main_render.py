from setup.voxel_setup import setup_voxel_scene
from common.plot import Plotter
from simulation.simulator import get_irrad_loc_dir, compute_ior_gradient
from simulation.simulate_utils import remove_under_floor

import taichi as ti
from scipy import ndimage
import numpy as np
# debug=True to check boundary access
ti.init(arch=ti.gpu)

scene, floor_height = setup_voxel_scene()

sampler_multiplier = 6
to_load_save = True
plotter = Plotter(sampler_multiplier, floor_height)

GAUSSIAN_SIGMA = 2.0
GAUSSIAN_RADIUS = 2

scene.ior = ndimage.gaussian_filter(scene.ior, sigma=GAUSSIAN_SIGMA, radius=GAUSSIAN_RADIUS)
scene.attenuation = ndimage.gaussian_filter(scene.attenuation, sigma=GAUSSIAN_SIGMA, radius=GAUSSIAN_RADIUS)
scene.scatter_strength = ndimage.gaussian_filter(scene.scatter_strength, sigma=GAUSSIAN_SIGMA, radius=GAUSSIAN_RADIUS)

scene.gradient = compute_ior_gradient(scene.ior)

scene.irradiance, scene.local_diretion = get_irrad_loc_dir(scene, sampler_multiplier, 
                                                           to_load_save=to_load_save, plotter=plotter)

# Post process the scene
scene.ior = remove_under_floor(scene.ior, floor_height)

ignorable_mask = np.linalg.norm(scene.gradient, axis=-1) < 0.05
temp_ior = scene.ior.copy()
temp_ior[ignorable_mask] = 1.0
scene.ior = temp_ior

temp_attenuation = scene.attenuation.copy()
temp_attenuation[ignorable_mask] = 0.0
scene.attenuation = temp_attenuation

temp_scatter_strength = scene.scatter_strength.copy()
temp_scatter_strength[ignorable_mask] = 0.0
scene.scatter_strength = temp_scatter_strength

temp_gradient = scene.gradient.copy()
temp_gradient[ignorable_mask, :] = 0
scene.gradient = temp_gradient

scene.display()