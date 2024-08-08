from setup.voxel_setup import setup_voxel_scene
from common.plot import Plotter
from simulation.simulator import get_irrad_loc_dir, compute_ior_gradient

import taichi as ti
from scipy import ndimage
import numpy as np
# debug=True to check boundary access
ti.init(arch=ti.gpu)

# "geometry", "bunny", "footed_glass", "stemmed_glass"
# scene_name = "bunny"

CONFIG = {
    "Name": "geometry",

    "Sampler Num": 5,
    "Gaus Sigma": 2.0,
    "Gaus Radius": 2,

    "Grad Threshold": 0.05,

    "Load Save": True,
}

scene, floor_height = setup_voxel_scene(CONFIG["Name"])

plotter = Plotter(CONFIG["Sampler Num"], floor_height)

scene.apply_filter(sigma=CONFIG["Gaus Sigma"], radius=CONFIG["Gaus Radius"])
scene.gradient = compute_ior_gradient(scene.ior)
scene.irradiance, scene.local_diretion = get_irrad_loc_dir(scene, CONFIG["Sampler Num"], 
                                                           to_load_save=CONFIG["Load Save"], plotter=plotter)

scene.truncate_outside_surface(gradient_threshold=CONFIG["Grad Threshold"]) # Post process the scene
scene.rt_render(translate_mode=True)