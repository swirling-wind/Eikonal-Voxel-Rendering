from setup.voxel_setup import setup_voxel_scene
from simulation.simulator import compute_ior_gradient

import taichi as ti
from scipy import ndimage

# debug=True to check boundary access
ti.init(arch=ti.gpu)
scene, floor_height = setup_voxel_scene()
scene.ior = ndimage.gaussian_filter(scene.ior, sigma=3.0, radius=1)
scene.gradient = compute_ior_gradient(scene.ior)
scene.display(ray_marching=True)