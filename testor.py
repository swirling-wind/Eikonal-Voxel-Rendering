from setup.voxel_setup import setup_voxel_scene
import taichi as ti
from simulation.simulator import compute_ior_gradient
from scipy import ndimage

# debug=True to check boundary access
ti.init(arch=ti.gpu)
NUM_XYZ = (128, 128, 128)
scene, floor_height = setup_voxel_scene(*NUM_XYZ)
scene.ior = ndimage.gaussian_filter(scene.ior, sigma=3.0, radius=1)
# scene.gradient = compute_ior_gradient(scene.ior)

scene.display(ray_marching=True)