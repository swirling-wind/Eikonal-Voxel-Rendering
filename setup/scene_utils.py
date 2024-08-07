import taichi as ti
import numpy as np

def setup_fields(voxel: np.ndarray, num_xyz: tuple[int, int, int]):
    voxel_field = ti.field(dtype=ti.u8, shape=num_xyz)
    voxel_field.from_numpy(voxel)
    return voxel_field

@ti.func
def origin_y(largest: int, r: int):
    return -(largest-r)-1

# def floor_ratio(largest: int) -> float:
#     return -1 / 64 * (largest)

def floor_height(num_y: int, floor_ratio: float) -> int:
    return int((1+floor_ratio) * num_y / 2)