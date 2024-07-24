import numpy as np
import matplotlib
import matplotlib.pyplot as plt


PLOT_STRIDE_LENGTH = 709

def floor_surface(num_x: int, num_y: int, floor_height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, zz = np.meshgrid(np.arange(num_x), np.arange(num_y))
    yy = np.ones_like(xx) * floor_height
    return xx, yy, zz

def plot_ior_field(ior_field: np.ndarray, initial_wavefront_positions: np.ndarray, initial_wavefront_directions: np.ndarray, sampler_multiplier: int, floor_height: int, num_shown_points: int=2000, ior_threshold: float=1.0):    
    plt.close("all") # clear previous plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.set_xlim(0, ior_field.shape[0])
    ax.set_ylim(0, ior_field.shape[1])
    ax.set_zlim(0, ior_field.shape[2])
    ax.set_title("Glass Sphere (blue color) inside a room with walls (red color) with a point light source at the top left corner of the room (green arrows)")
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    initial_pos_np = initial_wavefront_positions
    initial_dir_np = initial_wavefront_directions
    # Initial wavefront positions
    subsample_indices = np.arange(0, len(initial_pos_np), PLOT_STRIDE_LENGTH * sampler_multiplier)
    ax.quiver(initial_pos_np[subsample_indices, 0], initial_pos_np[subsample_indices, 1], initial_pos_np[subsample_indices, 2],
                initial_dir_np[subsample_indices, 0], initial_dir_np[subsample_indices, 1], initial_dir_np[subsample_indices, 2],
                color='green', length=1)

    # Regular sampling for the sphere
    x_points_sphere, y_points_sphere, z_points_sphere = np.where(ior_field > ior_threshold)
    subsample_step_sphere = max(1, len(x_points_sphere) // num_shown_points)  # Adjust the number to control the number of points
    subsample_indices_sphere = np.arange(0, len(x_points_sphere), subsample_step_sphere)

    ax.scatter(x_points_sphere[subsample_indices_sphere], y_points_sphere[subsample_indices_sphere], z_points_sphere[subsample_indices_sphere], color='blue', alpha=0.08)
    ax.view_init(elev=90, azim=-90)
    ax.plot_surface(*floor_surface(ior_field.shape[0], ior_field.shape[1], floor_height), color='red', alpha=0.5)
    plt.show()

def plot_gradients_3d(grad_xyz: np.ndarray, floor_height: int, threshold: float = 0.1, alpha: float = 0.5):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.set_xlim(0, grad_xyz.shape[0])
    ax.set_ylim(0, grad_xyz.shape[1])
    ax.set_zlim(0, grad_xyz.shape[2])
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    grad_magnitude = np.linalg.norm(grad_xyz, axis=-1)
    indices = np.where(grad_magnitude > threshold)
    ax.scatter(indices[0], indices[1], indices[2], c='red', alpha=alpha)
    ax.set_title('3D Visualization of Gradients')   
    # ax.set_box_aspect((np.ptp(indices[0]), np.ptp(indices[1]), np.ptp(indices[2])))  # Adjust the aspect ratio of the plot
    ax.view_init(elev=90, azim=-90)
    ax.plot_surface(*floor_surface(grad_xyz.shape[0], grad_xyz.shape[1], floor_height), color='red', alpha=0.5)
    plt.tight_layout()
    plt.show()