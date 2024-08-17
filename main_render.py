from setup.voxel_setup import setup_voxel_scene
from common.plot import Plotter
from simulation.simulator import get_irrad_loc_dir, compute_ior_gradient

import taichi as ti
from scipy import ndimage
import numpy as np

# ti.init(arch=ti.gpu, debug=True)
ti.init(arch=ti.gpu)

SCENE_CFG = {
    # Optional Names: "geometry" ## Temporarily not implemented and not needed: "bunny", "footed_glass", "stemmed_glass"
    "Name": "geometry", 
     
    # "HDR Res": (4000, 2000), 
    # "HDR Name": "Dark_night_4k.hdr",

    "HDR Res": (2000, 1000),
    "HDR Name": "Light_wooden_frame_room_2k.hdr",
    
    "NUM XYZ": (128, 128, 128),
    'Floor Ratio': -0.95,

    "Sampler Num": 6,

    "Load Save": True,
}

PROC_CFG = {
    "Gauss Sigma": 4.0,
    "Gauss Radius": 2,

    "Grad Threshold": 0.0,
}

plotter = Plotter(SCENE_CFG)
scene = setup_voxel_scene(SCENE_CFG)

scene.apply_filter(PROC_CFG)
scene.gradient = compute_ior_gradient(scene.ior)
scene.irradiance, scene.local_diretion = get_irrad_loc_dir(scene, SCENE_CFG, plotter=plotter)
scene.rt_render(free_mode=False)