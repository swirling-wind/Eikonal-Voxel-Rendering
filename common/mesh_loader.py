import numpy as np
from scipy import ndimage
import trimesh
import open3d as o3d

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