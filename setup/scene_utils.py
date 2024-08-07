import taichi as ti
import taichi.math as tm
import numpy as np

def setup_fields(voxel: np.ndarray, num_xyz: tuple[int, int, int]):
    voxel_field = ti.field(dtype=ti.u8, shape=num_xyz)
    voxel_field.from_numpy(voxel)
    return voxel_field

@ti.func
def origin_y(largest: int, r: int):
    return -(largest-r)-1

def get_floor_height(num_y: int, floor_ratio: float) -> int:
    return int((1+floor_ratio) * num_y / 2)



@ti.func
def ellipse(pos, x, y, z, r):
	return (x-pos[0])**2 + (y-pos[1])**2 + (z-pos[2])**2/2 - 3

@ti.func
def wineglass_range(x, y, z):
	return x**2 + y**2 - (ti.log(z+3.2))**2 - 0.02
    