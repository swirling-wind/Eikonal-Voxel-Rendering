import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

PLOT_STRIDE_LENGTH = 709
FIG_SIZE = (5, 5)

def floor_surface(num_x: int, num_y: int, floor_height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, zz = np.meshgrid(np.arange(num_x), np.arange(num_y))
    yy = np.ones_like(xx) * floor_height
    return xx, yy, zz

class Plotter:
    def __init__(self, sampler_multiplier: int, floor_height: int, ior_field: np.ndarray):
        self.sampler_multiplier = sampler_multiplier
        self.floor_height = floor_height
        self.ior_field = ior_field

    def plot_gradient(self, grad_xyz: np.ndarray, threshold: float = 0.1, alpha: float = 0.5):
        fig = plt.figure(figsize=FIG_SIZE)
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
        ax.view_init(elev=90, azim=-90)
        ax.plot_surface(*floor_surface(grad_xyz.shape[0], grad_xyz.shape[1], self.floor_height), color='red', alpha=0.5)
        plt.tight_layout()
        plt.show()

    
    def plot_wavefront_position(self, pos: np.ndarray, dir: np.ndarray, title: str, num_shown_points: int = 500):
        fig = plt.figure(figsize=FIG_SIZE)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1,1,1])
        ax.set_xlim(0, self.ior_field.shape[0])
        ax.set_ylim(0, self.ior_field.shape[1])
        ax.set_zlim(0, self.ior_field.shape[2])
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title(title)
        subsample_indices = np.arange(0, len(pos), PLOT_STRIDE_LENGTH * self.sampler_multiplier)
        ax.quiver(pos[subsample_indices, 0], pos[subsample_indices, 1], pos[subsample_indices, 2],
                    dir[subsample_indices, 0], dir[subsample_indices, 1], dir[subsample_indices, 2],
                    color='green', length=1)

        x_points, y_points, z_points = np.where(self.ior_field > 1.0)
        subsample_num = max(1, len(x_points) // num_shown_points)
        subsample_indices= np.arange(0, len(x_points), subsample_num)
        ax.scatter(x_points[subsample_indices], y_points[subsample_indices], z_points[subsample_indices], color='blue', alpha=0.1)
        ax.view_init(elev=90, azim=-90)
        ax.plot_surface(*floor_surface(self.ior_field.shape[0], self.ior_field.shape[1], self.floor_height), color='red', alpha=0.5)
        plt.show()

    def plot_irradiance_grid(self, radiometric_grid: np.ndarray, threshold=3.0):
        fig = plt.figure(figsize=FIG_SIZE)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(0, radiometric_grid.shape[0])
        ax.set_ylim(0, radiometric_grid.shape[1])
        ax.set_zlim(0, radiometric_grid.shape[2])
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('Radiometric Grid in 3D Projection')
        ax.view_init(elev=90, azim=-90)
        ax.plot_surface(*floor_surface(radiometric_grid.shape[0], radiometric_grid.shape[1], self.floor_height), color='red', alpha=0.5)
        x_points, y_points, z_points = np.where(radiometric_grid > threshold)
        assert len(x_points) > 0, "No points found above the threshold value"
        radiometric_grid_filtered = radiometric_grid[x_points, y_points, z_points]

        ax.scatter(x_points, y_points, z_points, c=radiometric_grid_filtered, cmap='hot', alpha=0.02)    
        norm = Normalize(vmin=np.min(radiometric_grid_filtered), vmax=np.max(radiometric_grid_filtered)) # Normalize the colorbar
        mappable = ScalarMappable(norm=norm, cmap='hot')
        mappable.set_array(radiometric_grid_filtered)
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)  # Add colorbar
        plt.show()

    def plot_irradiance_slices(self, radiometric_grid: np.ndarray, threshold=3.0, num_slices=8, z_start=None, z_end=None):  
        if z_start is None:
            z_start = 0
        if z_end is None:
            z_end = radiometric_grid.shape[2] - 1

        z_indices = np.linspace(z_start, z_end, num_slices, dtype=int)
        num_rows = (num_slices + 1) // 2
        num_cols = 2    
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 3 * num_rows))
        axes = axes.flatten()    
        for i, z_index in enumerate(z_indices):
            ax = axes[i]
            
            slice_data = radiometric_grid[:, :, z_index].T
            slice_data[slice_data < threshold] = 0
            
            im = ax.imshow(slice_data, cmap='hot', origin='lower')
            ax.set_title(f"Irradiance at Z = {z_index}")
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            
            fig.colorbar(im, ax=ax, shrink=0.8)

        for ax in axes[len(z_indices):]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        