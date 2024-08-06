import taichi as ti
import numpy as np

def setup_fields(bunny_voxels: np.ndarray, glass_voxels: np.ndarray, num_xyz: tuple[int, int, int]) -> tuple:
    bunny_field = ti.field(dtype=ti.u8, shape=num_xyz)
    bunny_field.from_numpy(bunny_voxels)
    glass_field = ti.field(dtype=ti.u8, shape=num_xyz)
    glass_field.from_numpy(glass_voxels)
    return bunny_field, glass_field

@ti.func
def origin_y(largest: int, r: int):
    return -(largest-r)-1

# def floor_ratio(largest: int) -> float:
#     return -1 / 64 * (largest)

def floor_height(num_y: int, floor_ratio: float) -> int:
    return int((1+floor_ratio) * num_y / 2)