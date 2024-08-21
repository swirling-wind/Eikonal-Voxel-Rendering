import numpy as np
from scipy import ndimage
import trimesh
import open3d as o3d

def load_and_voxelize_mesh(file_path: str, num_xyz: tuple[int, int, int], 
                           voxel_size=0.005, need_rotate=False) -> np.ndarray:
    """
    Smaller the voxel_size, higher the resolution of the voxel grid.
    """
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


def load_and_project_mesh(file_path: str, ior: float, resolution: tuple[int, int], 
                          voxel_size=0.005) -> np.ndarray:
    """
    Load a 3D mesh and project it onto the XY plane, creating a 2D numpy array.
    
    :param file_path: Path to the 3D mesh file
    :param resolution: Tuple (width, height) for the output 2D array
    :param voxel_size: Size of voxels for initial voxelization (smaller = higher resolution)
    :return: 2D numpy array representing the projection
    """
    # Load the mesh
    target_mesh = trimesh.load(file_path)
    assert isinstance(target_mesh, trimesh.Trimesh), "Loaded object should be a Trimesh"
    
    # Convert to Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(target_mesh.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(target_mesh.faces)
    mesh.compute_vertex_normals()
    
    # Create voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
    
    # Initialize the 2D projection array
    projection = np.ones(resolution, dtype=np.float32)
    
    # Project voxels onto XY plane
    for voxel in voxel_grid.get_voxels():
        x, y, _ = voxel.grid_index
        if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
            projection[x, y] = ior
    
    print("Projected shape:", projection.shape, " from:", file_path)
    print("Number of filled pixels:", np.sum(projection))
    
    return projection