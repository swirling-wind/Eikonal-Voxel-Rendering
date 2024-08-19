from common.figure import *
from setup.scene_utils import get_floor_height

from data.mlp import MLPFitter
import matplotlib.pyplot as plt
import numpy as np

SCENE_CFG = {
    "Name": "geometry", 

    "HDR Res": (2000, 1000),
    "HDR Name": "Light_wooden_frame_room_2k.hdr",
    "Cam Pos": [(2, 0.5, 2), (-2, -1, 2), (-2, -1, 0)],

    "Screen Res": (1280, 960),
    
    "Num XYZ": (128, 128, 128),
    'Floor Ratio': -0.95,

    "Sampler Num": 8,

    "Load Save": True,

    "Save Fig": False,
}

floor_height = get_floor_height(SCENE_CFG["Num XYZ"][1], SCENE_CFG["Floor Ratio"])

mlp = MLPFitter(np.zeros((128,128,128)), SCENE_CFG, num_epoches=1000)
mlp_predicted_irradiance = mlp.predict(pad=True)
# plotter.plot_irradiance_slices(mlp_predicted_irradiance, "mlp-irrad", threshold=3, 
#                                num_slices=4, z_start=30, z_end=100)
print(mlp_predicted_irradiance.shape)

# When training, y in [0, 125]. x, z in [0, 128]
# x, y, z = 30, 90, 60
x, y, z = 110, 90, 60
print(mlp_predicted_irradiance[x, y + floor_height, z])
print(mlp.query([x, y, z]))