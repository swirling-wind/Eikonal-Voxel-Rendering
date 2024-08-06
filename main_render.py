from setup.voxel_setup import setup_voxel_scene
from common.plot import Plotter
from simulation.simulator import get_irrad_loc_dir, compute_ior_gradient
from simulation.simulate_utils import remove_under_floor

import taichi as ti
from scipy import ndimage
# debug=True to check boundary access
ti.init(arch=ti.gpu)

scene, floor_height = setup_voxel_scene()

sampler_multiplier = 7
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
scene.ior = remove_under_floor(scene.ior, floor_height)

scene.display()