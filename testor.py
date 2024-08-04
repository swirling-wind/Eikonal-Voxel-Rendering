from setup.voxel_setup import setup_voxel_scene
from common.plot import Plotter
from simulation.simulator import get_irrad_loc_dir, compute_ior_gradient

import taichi as ti
from scipy import ndimage
# debug=True to check boundary access
ti.init(arch=ti.gpu)

scene, floor_height = setup_voxel_scene()
print("Scene setup done. Starting simulation...")
sampler_multiplier = 7
to_load_save = True
plotter = Plotter(sampler_multiplier, floor_height)
scene.ior = ndimage.gaussian_filter(scene.ior, sigma=3.0, radius=1)
scene.gradient = compute_ior_gradient(scene.ior)
scene.irradiance, scene.local_diretion = get_irrad_loc_dir(scene, sampler_multiplier, to_load_save=to_load_save, plotter=plotter)

scene.display()