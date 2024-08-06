from setup.voxel_setup import setup_voxel_scene
from common.plot import Plotter
from simulation.simulator import get_irrad_loc_dir, compute_ior_gradient

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

scene.apply_filter(sigma=GAUSSIAN_SIGMA, radius=GAUSSIAN_RADIUS)
scene.gradient = compute_ior_gradient(scene.ior)
scene.irradiance, scene.local_diretion = get_irrad_loc_dir(scene, sampler_multiplier, 
                                                           to_load_save=to_load_save, plotter=plotter)

scene.truncate_outside_surface(gradient_threshold=0.06) # Post process the scene
scene.display()