from .scene import Scene
from common.mesh_loader import load_and_voxelize_mesh

import taichi as ti
import taichi.math as tm
import numpy as np

NUM_XYZ = (128, 128, 128)
GLASS_IOR = 1.5
LARGE_R, CUBE_LEN = 23, 30

RED, BLUE, GREY = tm.vec3(0.9, 0, 0.1), tm.vec3(0, 0.5, 1), tm.vec3(0.7, 0.7, 0.7), 
WHITE, AZURE = tm.vec3(1, 1, 1), tm.vec3(0.4, 0.7, 1)


def setup_fields(bunny_voxels: np.ndarray, glass_voxels: np.ndarray, num_xyz: tuple[int, int, int]) -> tuple:
    bunny_field = ti.field(dtype=ti.u8, shape=num_xyz)
    bunny_field.from_numpy(bunny_voxels)
    glass_field = ti.field(dtype=ti.u8, shape=num_xyz)
    glass_field.from_numpy(glass_voxels)
    return bunny_field, glass_field

@ti.func
def origin_y(largest: int, r: int):
    return -(largest-r)-1

def floor_ratio(largest: int) -> float:
    return -1 / 64 * (largest)

def floor_height(num_y: int, floor_ratio: float) -> int:
    return int((1+floor_ratio) * num_y / 2)

@ti.func
def add_ball(r: ti.i32, origin: tm.vec3, mat: ti.i8, 
             color: tm.vec3, voxel_ior: float):
    pad = 1
    for i, j, k in ti.ndrange((-r-pad, r+pad), (-r-pad, r+pad), (-r-pad, r+pad)):
        xyz = tm.ivec3(i, j, k)
        if xyz.dot(xyz) < r**2: 
            scene.set_voxel(tm.vec3(i, j, k), origin, mat, color, ior=voxel_ior)
            scene.set_voxel_data(tm.vec3(i, j, k), origin, atten=0.02, scatter_strength=0.5,
                                  anisotropy_factor=0.0, opaque=0)
            
@ti.func
def add_cube(side_len: int, left_bottom_corner: tm.vec3, mat: ti.i8, 
             color: tm.vec3, voxel_ior: float):
    for i, j, k in ti.ndrange(side_len, side_len, side_len):
        scene.set_voxel(tm.vec3(i, j, k), left_bottom_corner, mat, color, ior=voxel_ior)
        scene.set_voxel_data(tm.vec3(i, j, k), left_bottom_corner, atten=0.02, scatter_strength=0.5,
                             anisotropy_factor=0.0, opaque=0)

@ti.func
def add_glass(glass_field, origin: tm.vec3, mat: ti.i8, 
              color: tm.vec3, voxel_ior: float, num_x: int, num_y: int, num_z: int):
    for i, j, k in ti.ndrange(num_x, num_y, num_z):
        if glass_field[i, j, k] == 1:
            scene.set_voxel(tm.vec3(i, j, k), origin, mat, color, ior=voxel_ior)
            scene.set_voxel_data(tm.vec3(i, j, k), origin, atten=0.02, scatter_strength=0.5,
                                  anisotropy_factor=0.0, opaque=0)

@ti.func 
def add_bunny(bunny_field, origin: tm.vec3, mat: ti.i8, 
              color: tm.vec3, voxel_ior: float, num_x: int, num_y: int, num_z: int):
    for i, j, k in ti.ndrange(num_x, num_y, num_z):
        if bunny_field[i, j, k] == 1:
            scene.set_voxel(tm.vec3(i, j, k), origin, mat, color, ior=voxel_ior)
            scene.set_voxel_data(tm.vec3(i, j, k), origin, atten=0.02, scatter_strength=0.5,
                                  anisotropy_factor=0.0, opaque=0)

@ti.kernel
def initialize_voxels(bunny_field: ti.template(), glass_field: ti.template(), floor_ratio: float, num_x: int, num_y: int, num_z: int): # type: ignore
    add_ball(LARGE_R, tm.vec3(-34, floor_ratio * 64 / 2, 20), 1, RED, GLASS_IOR)
    add_cube(CUBE_LEN, tm.vec3(-52, floor_ratio * 64 + 2, -46), 1, BLUE, GLASS_IOR)
    add_glass(glass_field, tm.vec3(-16, floor_ratio * 64 + 1, -110), 1, WHITE, GLASS_IOR, num_x, num_y, num_z) # coordinate z must be minus, because of the potential index out of range of the voxel field
    add_bunny(bunny_field, tm.vec3(-4, floor_ratio * 64 + 2, 10), 1, GREY, GLASS_IOR, num_x, num_y, num_z)
    
    # for debugging
    # scene.set_voxel(tm.vec3(0,0,0), tm.vec3(0,20,0), 1, RED, GLASS_IOR)

def setup_voxel_scene() -> tuple[Scene, int]:
    global scene, bunny_field, glass_field

    num_x, num_y, num_z = NUM_XYZ
    bunny_voxels = load_and_voxelize_mesh("./assets/bun_zipper_res3.ply", NUM_XYZ, 0.0025)
    glass_voxels = load_and_voxelize_mesh("./assets/wine_glass.obj", NUM_XYZ, 0.040, need_rotate=True)
    bunny_field, glass_field = setup_fields(bunny_voxels, glass_voxels, NUM_XYZ)
    
    floor_ratio_val = -0.9
    print("Floor Ratio:", floor_ratio_val, ", Floor Height:", floor_height(num_y, floor_ratio_val))
    
    scene = Scene(exposure=1.0)
    scene.set_directional_light((0, 1, 0), 0.2, (1, 1, 1))
    scene.set_background_color((0.05, 0.05, 0.4))
    scene.set_floor(height=floor_ratio_val, color=tm.vec3(0.2, 0.2, 0.2))

    initialize_voxels(bunny_field, glass_field, floor_ratio_val, num_x, num_y, num_z)
    return scene, floor_height(num_y, floor_ratio_val)
