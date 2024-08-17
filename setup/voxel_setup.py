from .scene import Scene
from .scene_utils import *
from common.mesh_loader import load_and_voxelize_mesh

import taichi as ti
import taichi.math as tm
import numpy as np

GLASS_IOR = 1.5
RED, BLUE, GREY, WHITE, AZURE = tm.vec3(0.9, 0, 0.1), tm.vec3(0, 0.5, 1), tm.vec3(0.7, 0.7, 0.7), tm.vec3(1, 1, 1), tm.vec3(0.4, 0.7, 1)

ATTENUATION = 0.01
############### Add different objects ###############
@ti.func
def add_ball(r: ti.i32, origin: tm.vec3, mat: ti.i8, 
             color: tm.vec3, voxel_ior: float):
    pad = 1
    for i, j, k in ti.ndrange((-r-pad, r+pad), (-r-pad, r+pad), (-r-pad, r+pad)):
        xyz = tm.ivec3(i, j, k)
        if xyz.dot(xyz) < r**2: 
            scene.set_voxel(tm.vec3(i, j, k), origin, mat, color, ior=voxel_ior)
            scene.set_voxel_data(tm.vec3(i, j, k), origin, atten=ATTENUATION, scatter_strength=0.5)
            
@ti.func
def add_cube(side_len: int, left_bottom_corner: tm.vec3, mat: ti.i8, 
             color: tm.vec3, voxel_ior: float):
    for i, j, k in ti.ndrange(side_len, side_len, side_len):
        scene.set_voxel(tm.vec3(i, j, k), left_bottom_corner, mat, color, ior=voxel_ior)
        scene.set_voxel_data(tm.vec3(i, j, k), left_bottom_corner, atten=ATTENUATION, scatter_strength=0.5)

@ti.func
def add_stemmed_glass(glass_field, origin: tm.vec3, mat: ti.i8, 
              color: tm.vec3, voxel_ior: float, num_x: int, num_y: int, num_z: int):
    for i, j, k in ti.ndrange(num_x, num_y, num_z):
        if glass_field[i, j, k] == 1:
            scene.set_voxel(tm.vec3(i, j, k), origin, mat, color, ior=voxel_ior)
            scene.set_voxel_data(tm.vec3(i, j, k), origin, atten=ATTENUATION, scatter_strength=0.5)

@ti.func 
def add_bunny(bunny_field, origin: tm.vec3, mat: ti.i8, 
              color: tm.vec3, voxel_ior: float, num_x: int, num_y: int, num_z: int):
    for i, j, k in ti.ndrange(num_x, num_y, num_z):
        if bunny_field[i, j, k] == 1:
            scene.set_voxel(tm.vec3(i, j, k), origin, mat, color, ior=voxel_ior)
            scene.set_voxel_data(tm.vec3(i, j, k), origin, atten=ATTENUATION, scatter_strength=0.5)
            
############### Init different scenes ###############
@ti.kernel
def init_geometry(floor_ratio: float):
    LARGE_R, CUBE_LEN = 50, 35
    add_ball(LARGE_R, tm.vec3(0, floor_ratio * 64 / 2 + 26, 0), 1, RED, GLASS_IOR)
    add_cube(CUBE_LEN, tm.vec3(-50, floor_ratio * 64 + 1, -46), 1, BLUE, GLASS_IOR)

@ti.kernel
def init_stemmed_glass(glass_field: ti.template(), floor_ratio: float, num_x: int, num_y: int, num_z: int):
    # coordinate z must be minus, because of the potential index out of range of the voxel field
    add_stemmed_glass(glass_field, tm.vec3(-16, floor_ratio * 64 + 2, -110), 1, WHITE, GLASS_IOR, num_x, num_y, num_z)

@ti.kernel
def init_bunny(bunny_field: ti.template(), floor_ratio: float, num_x: int, num_y: int, num_z: int): # type: ignore
    add_bunny(bunny_field, tm.vec3(-50, floor_ratio * 64 + 7, -40), 1, GREY, GLASS_IOR, num_x, num_y, num_z)

# @ti.kernel
# def init_footed_glass(height: int):
#     N = height
#     color = tm.vec3(0, 0.3, 0.3)
#     for i, j, k in ti.ndrange((-N, N), (-N, N // 2), (-N, N)):
#         x, y, z = float(i) / (N/3), float(j) / (N/3), float(k) / (N/3)
#         if wineglass_range(x, y, z) <= 0 and ellipse(tm.vec3(0, 0, 3.0), x, y, z, 1) >= 0:
#             scene.set_voxel(tm.vec3(i, k, j), tm.vec3(0, 0, 0), 1, color, ior=GLASS_IOR)
#             scene.set_voxel_data(tm.vec3(i, k, j), tm.vec3(0, 0, 0), atten=0.03, scatter_strength=0.5,
#                                  anisotropy_factor=0.0, opaque=0)

# @ti.kernel
# def init_ceiling_light(num_x: int, num_y: int, num_z: int):
#     ceiling_y = num_y // 2 - 1
#     # fill the toppest layer of x-z plane with light
#     for x, z in ti.ndrange((-num_x // 2 + 1, num_x // 2 - 1), (-num_z // 2 + 1, num_z // 2 - 1)):
#         scene.set_voxel(tm.vec3(x, ceiling_y, z), tm.vec3(0), 1, WHITE, GLASS_IOR)
#         scene.set_voxel_data(tm.vec3(x, ceiling_y, z), tm.vec3(0), atten=0.03, scatter_strength=0.5)
#         scene.set_voxel_emit(tm.vec3(x, ceiling_y, z), tm.vec3(0), WHITE)

@ti.kernel
def init_debug_voxel():
    scene.set_voxel(tm.vec3(0,-60,0), tm.vec3(0,0,0), 1, RED, 1.0)
    

def setup_voxel_scene(scene_config: dict) -> Scene:
    global scene

    NUM_XYZ = scene_config['Num XYZ']
    num_x, num_y, num_z = NUM_XYZ
    floor_ratio_val = scene_config['Floor Ratio']
    load_scene = scene_config["Name"]
    
    floor_height = get_floor_height(num_y, floor_ratio_val)
    print("Floor Ratio:", floor_ratio_val, ", Floor Height:", floor_height)
    
    scene = Scene(scene_config["HDR Res"], scene_config["HDR Name"], scene_config["Screen Res"], exposure=1.0)
    scene.set_directional_light((0, 1, 0), 0.2, (1, 1, 1))
    scene.set_background_color((0.05, 0.05, 0.4))
    scene.set_floor(height=floor_ratio_val, color=tm.vec3(0.2, 0.01, 0.01))
    
    if load_scene == 'geometry':
        init_geometry(floor_ratio_val)
    elif load_scene == 'bunny':
        bunny_voxel = load_and_voxelize_mesh("./assets/bun_zipper_res3.ply", NUM_XYZ, 0.0016)
        bunny_field = setup_fields(bunny_voxel, NUM_XYZ)
        init_bunny(bunny_field, floor_ratio_val, num_x, num_y, num_z)
    # elif load_scene == 'footed_glass':
    #     init_footed_glass(40)
    # elif load_scene == 'stemmed_glass':
    #     stemmed_glass_voxel = load_and_voxelize_mesh("./assets/wine_glass.obj", NUM_XYZ, 0.040, need_rotate=True)
    #     stemmed_glass_field = setup_fields(stemmed_glass_voxel, NUM_XYZ)
    #     init_stemmed_glass(stemmed_glass_field, floor_ratio_val, num_x, num_y, num_z)
    
    return scene
