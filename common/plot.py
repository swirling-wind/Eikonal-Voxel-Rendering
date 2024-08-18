import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
from setup.scene_utils import get_floor_height

PLOT_STRIDE_LENGTH = 439 # 439# 293 # 709 
FIG_SIZE = (5, 5)

def floor_surface(num_x: int, num_y: int, floor_height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, zz = np.meshgrid(np.arange(num_x), np.arange(num_y))
    yy = np.ones_like(xx) * floor_height
    return xx, yy, zz

class Plotter:
    def __init__(self, scene_config: dict):
        self.sampler_multiplier = scene_config['Sampler Num']
        self.scene_name = scene_config['Name']
        self.hdr_name = scene_config['HDR Name']
        self.floor_height = get_floor_height(scene_config['Num XYZ'][1], scene_config['Floor Ratio'])

    def plot_gradient(self, grad_xyz: np.ndarray, threshold: float = 0.1, alpha: float = 0.05):
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
        subsample_indices = np.arange(0, len(indices[0]), PLOT_STRIDE_LENGTH // 16)
        ax.scatter(indices[0][subsample_indices], indices[1][subsample_indices], indices[2][subsample_indices], color='blue', alpha=alpha)
        ax.set_title('3D Visualization of Gradients')
        ax.view_init(elev=90, azim=-90)
        ax.plot_surface(*floor_surface(grad_xyz.shape[0], grad_xyz.shape[1], self.floor_height), color='red', alpha=0.5)
        plt.tight_layout()
        plt.show()

    
    def plot_wavefront(self, ior_field: np.ndarray, pos: np.ndarray|None, dir: np.ndarray|None, 
                       title: str = "IOR and wavefront", num_shown_points: int = 1000):
        fig = plt.figure(figsize=FIG_SIZE)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1,1,1])
        ax.set_xlim(0, ior_field.shape[0])
        ax.set_ylim(0, ior_field.shape[1])
        ax.set_zlim(0, ior_field.shape[2])
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title(title)
        if pos is not None and dir is not None:
            subsample_indices = np.arange(0, len(pos), PLOT_STRIDE_LENGTH * self.sampler_multiplier)
            ax.quiver(pos[subsample_indices, 0], pos[subsample_indices, 1], pos[subsample_indices, 2],
                        dir[subsample_indices, 0], dir[subsample_indices, 1], dir[subsample_indices, 2],
                        color='green', length=1)
        x_points, y_points, z_points = np.where(ior_field > 1.0)
        subsample_num = max(1, len(x_points) // num_shown_points)
        subsample_indices= np.arange(0, len(x_points), subsample_num)
        ax.scatter(x_points[subsample_indices], y_points[subsample_indices], z_points[subsample_indices], color='blue', alpha=0.1)
        ax.view_init(elev=90, azim=-90)
        ax.plot_surface(*floor_surface(ior_field.shape[0], ior_field.shape[1], self.floor_height), color='red', alpha=0.5)
        plt.show()

    def plot_irradiance_grid(self, radiometric_grid: np.ndarray, threshold=40.0):
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

    def plot_irradiance_slices(self, radiometric_grid: np.ndarray, title: str, threshold=3.0, num_slices=4, z_start=None, z_end=None):  
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

        fig_path = os.path.join(os.getcwd(), 'images', self.hdr_name, self.scene_name, title)
        plt.savefig(fig_path + '.png', dpi=300)
        plt.show()
        

    def visualize_local_direction_grid_slices(self, direction_grid, num_slices=8, z_start=None, z_end=None):
        if z_start is None:
            z_start = 0
        if z_end is None:
            z_end = direction_grid.shape[2] - 1
        
        z_indices = np.linspace(z_start, z_end, num_slices, dtype=int)
        num_rows = (num_slices + 1) // 2
        num_cols = 2
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
        axes = axes.flatten()
        
        for i, z_index in enumerate(z_indices):
            ax = axes[i]
            
            slice_data = np.transpose(direction_grid[:, :, z_index], axes=(1, 0, 2))
            y, x = np.meshgrid(np.arange(slice_data.shape[0]), np.arange(slice_data.shape[1]), indexing='ij')
            u, v = slice_data[..., 0], slice_data[..., 1]
            
            ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=0.9)
            ax.set_title(f"Local Direction at Z = {z_index}")
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.set_aspect('equal')
        
        for ax in axes[len(z_indices):]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    def plot_local_direction_grid_slices(self, direction_grid, num_slices=8, z_start=None, z_end=None, stream_plot=True, subsample_rate=3):
        if z_start is None:
            z_start = 0
        if z_end is None:
            z_end = direction_grid.shape[2] - 1
        
        z_indices = np.linspace(z_start, z_end, num_slices, dtype=int)
        num_rows = (num_slices + 1) // 2
        num_cols = 2
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten()
        
        for i, z_index in enumerate(z_indices):
            ax = axes[i]
            
            slice_data = np.transpose(direction_grid[:, :, z_index], axes=(1, 0, 2))
            y, x = np.meshgrid(np.arange(slice_data.shape[0]), np.arange(slice_data.shape[1]), indexing='ij')
            u, v = slice_data[..., 0], slice_data[..., 1]
            
            # 降采样
            x_subsampled = x[::subsample_rate, ::subsample_rate]
            y_subsampled = y[::subsample_rate, ::subsample_rate]
            u_subsampled = u[::subsample_rate, ::subsample_rate]
            v_subsampled = v[::subsample_rate, ::subsample_rate]
            
            # 绘制箭头
            ax.quiver(x_subsampled, y_subsampled, u_subsampled, v_subsampled, angles='xy', scale_units='xy', scale=0.5)
            
            # 绘制流线图
            if stream_plot:
                ax.streamplot(x, y, u, v, density=1, linewidth=0.5, color='gray', arrowstyle='->')
            
            ax.set_title(f"Local Direction at Z = {z_index}")
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.set_aspect('equal')
        
        for ax in axes[len(z_indices):]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()