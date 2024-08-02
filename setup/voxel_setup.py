from .scene import Scene
import taichi as ti
# from taichi.math import *
import taichi.math as tm

import numpy as np
from scipy import ndimage
import trimesh
import open3d as o3d

GLASS_IOR = 1.5
LARGE_R, MEDIUM_R = 20, 15
RED, BLUE, GREY = tm.vec3(0.9, 0, 0.1), tm.vec3(0, 0.5, 1), tm.vec3(0.7, 0.7, 0.7), 
WHITE, AZURE = tm.vec3(1, 1, 1), tm.vec3(0.4, 0.7, 1)

def load_and_voxelize_mesh(file_path: str, num_xyz: tuple[int, int, int], 
                           voxel_size=0.005, need_rotate=False) -> np.ndarray:
    target_mesh = trimesh.load(file_path)
    assert isinstance(target_mesh, trimesh.Trimesh), "Loaded object should be a Trimesh"
    vertices, faces = target_mesh.vertices, target_mesh.faces

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)

    filled_voxels = np.zeros(num_xyz, dtype=np.uint8)
    for voxel in voxel_grid.get_voxels():
        voxel_coord = voxel.grid_index
        filled_voxels[voxel_coord[0], voxel_coord[1], voxel_coord[2]] = True
    for y in range(filled_voxels.shape[1]):
        slice_y = filled_voxels[:, y, :]
        filled_slice = ndimage.binary_fill_holes(slice_y)
        filled_voxels[:, y, :] = filled_slice

    if need_rotate:
        filled_voxels = np.transpose(filled_voxels, (0, 2, 1))  # Transpose from (x, y, z) to (x, z, y)
        filled_voxels = np.flip(filled_voxels, axis=2)  # Flip y-axis to make the object stand up
    print("Loaded Voxel shape:", filled_voxels.shape, " from:", file_path)  
    print("Number of filled voxels:", np.sum(filled_voxels))
    return filled_voxels

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

@ti.kernel
def initialize_voxels(bunny_field: ti.template(), glass_field: ti.template(), floor_ratio: float, num_x: int, num_y: int, num_z: int): # type: ignore
    add_ball(LARGE_R, tm.vec3(-32, origin_y(LARGE_R, LARGE_R), 0), 1, RED, GLASS_IOR)
    add_ball(MEDIUM_R, tm.vec3(-8, origin_y(LARGE_R, MEDIUM_R), 36), 1, BLUE, GLASS_IOR)
    add_glass(glass_field, tm.vec3(-24, floor_ratio * 64, -128), 1, WHITE, GLASS_IOR, num_x, num_y, num_z) # coordinate z must be minus, because of the potential index out of range of the voxel field
    add_bunny(bunny_field, tm.vec3(3, floor_ratio * 64, 0), 1, GREY, GLASS_IOR, num_x, num_y, num_z)
    
    scene.set_voxel(tm.vec3(0,0,0), tm.vec3(0,20,0), 1, RED, GLASS_IOR)

def setup_voxel_scene(num_x: int, num_y: int, num_z: int) -> tuple[Scene, int]:
    global scene, bunny_field, glass_field

    num_xyz = (num_x, num_y, num_z)
    bunny_voxels = load_and_voxelize_mesh("./assets/bun_zipper_res4.ply", num_xyz, 0.004)
    glass_voxels = load_and_voxelize_mesh("./assets/wine_glass.obj", num_xyz, 0.07, need_rotate=True)
    bunny_field, glass_field = setup_fields(bunny_voxels, glass_voxels, num_xyz)

    scene = Scene(exposure=1.2)
    scene.set_directional_light((0, 1, 0), 0.2, (1, 1, 1))
    scene.set_background_color((1, 0.9, 0.9))

    floor_ratio_val = floor_ratio(LARGE_R)
    print("Floor Ratio:", floor_ratio_val, ", Floor Height:", floor_height(num_y, floor_ratio_val))
    scene.set_floor(height=floor_ratio_val, color=AZURE)

    initialize_voxels(bunny_field, glass_field, floor_ratio_val, num_x, num_y, num_z)
    return scene, floor_height(num_y, floor_ratio_val)
