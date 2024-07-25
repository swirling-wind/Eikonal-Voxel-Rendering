from common.plot import Plotter

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

def generate_initial_wavefront(num_samplers_per_voxel : int, pos_perturbation_scale : float, num_x=128, num_y=128, num_z=128) -> tuple[np.ndarray, np.ndarray]:
    sampler_multiplier = num_samplers_per_voxel
    position_perturbation = np.random.uniform(-pos_perturbation_scale, pos_perturbation_scale, (num_x * num_z * (sampler_multiplier**3), 3))
    initial_wavefront_pos = np.array([(x / sampler_multiplier, y / sampler_multiplier + num_y - 1.5, z / sampler_multiplier) 
                                      for x in range(num_x * sampler_multiplier) 
                                      for y in range(sampler_multiplier) 
                                      for z in range(num_z * sampler_multiplier)]) + position_perturbation
    initial_wavefront_dir = np.array([(0, -1, 0) for _ in range(num_x * num_z * (sampler_multiplier**3))])
    return initial_wavefront_pos, initial_wavefront_dir

def compute_ior_gradient(ior_field: np.ndarray) -> np.ndarray:
    grad_xyz = np.gradient(ior_field)
    return np.stack(grad_xyz, axis=-1)

@torch.jit.script
def update_wavefront(pos: torch.Tensor, dir: torch.Tensor, within_mask: torch.Tensor, grad_xyz: torch.Tensor, IOR: torch.Tensor, step_size: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = pos.int()
    # Obtain the refractive index at the current position
    n = IOR[indices[:, 0].clamp(0, IOR.shape[0] - 1),
            indices[:, 1].clamp(0, IOR.shape[1] - 1),
            indices[:, 2].clamp(0, IOR.shape[2] - 1)]

    # Calculate the new position and direction of the wavefront
    new_pos = pos + step_size * dir / (n**2).unsqueeze(1)

    # Create a mask to identify the wavefront positions that are within the boundaries of the IOR field
    new_within_mask = (new_pos[:, 0] >= 0) & (new_pos[:, 0] < IOR.shape[0]) & \
                      (new_pos[:, 1] >= 0) & (new_pos[:, 1] < IOR.shape[1]) & \
                      (new_pos[:, 2] >= 0) & (new_pos[:, 2] < IOR.shape[2]) & \
                      within_mask
    
    # Clamp the indices to ensure they are within the valid range of grad_xyz
    clamped_indices_x = indices[:, 0].clamp(0, grad_xyz.shape[0] - 1)
    clamped_indices_y = indices[:, 1].clamp(0, grad_xyz.shape[1] - 1)
    clamped_indices_z = indices[:, 2].clamp(0, grad_xyz.shape[2] - 1)

    # Update the direction only for the wavefront positions within the IOR boundaries
    new_dir = torch.where(new_within_mask.unsqueeze(1),
                          dir + step_size * grad_xyz[clamped_indices_x, clamped_indices_y, clamped_indices_z] / n.unsqueeze(1),
                          dir)
    return new_pos, new_dir, new_within_mask


def simulate_wavefront_propagation(ior_field: np.ndarray, grad_xyz: np.ndarray, atten_grid: np.ndarray,
                                   initial_wavefront_pos: np.ndarray, initial_wavefront_dir: np.ndarray, 
                                   plotter: Plotter,
                                   num_steps: int = 100, step_size: float = 1.0, num_show_images: int = 3) -> np.ndarray:
    
    stride = max(num_steps // (num_show_images-1), 1)
    plot_step_indices =  [i for i in range(stride, num_steps+1, stride)] + [num_steps - 1] if num_show_images > 0 else []

    # Initialize the wavefront position and direction (initial_wavefront_pos.shape[0] = 30000 or highter) and the mask to keep track of the wavefront positions within the IOR boundaries
    cur_pos = torch.tensor(initial_wavefront_pos, device=DEVICE)
    cur_dir = torch.tensor(initial_wavefront_dir, device=DEVICE)
    within_mask = torch.ones(initial_wavefront_pos.shape[0], dtype=torch.bool, device=DEVICE)

    # Initialize the all photons' absorption A to zero
    photon_energy = torch.ones(initial_wavefront_pos.shape[0], device=DEVICE)

    # Convert (128,128,128) voxel grids to tensors
    voxel_ior = torch.tensor(ior_field, device=DEVICE)
    voxel_grad = torch.tensor(grad_xyz, device=DEVICE)
    voxel_atten = torch.tensor(atten_grid, device=DEVICE)

    irradiance_grid = torch.zeros(ior_field.shape, device=DEVICE)

    for cur_step in range(num_steps):
        new_positions, new_directions, within_mask = update_wavefront(cur_pos, cur_dir, within_mask, voxel_grad, voxel_ior, step_size)
        cur_pos = new_positions
        cur_dir = new_directions

        within_indices = cur_pos.int()[within_mask]

        attenuation_coef = voxel_atten[within_indices[:, 0], within_indices[:, 1], within_indices[:, 2]]
        photon_energy[within_mask] *= torch.exp(-attenuation_coef)

        # - Accumulate each attenuated photon energy to its corresponding voxel in the irradiance grid. The inverse_indices is used to sum the energy of the same photon that is scattered to multiple voxels
        unique_indices, inverse_indices = torch.unique(within_indices, return_inverse=True, dim=0)
        energy_sum = torch.zeros(unique_indices.shape[0], device=DEVICE)
        energy_sum.scatter_add_(0, inverse_indices, photon_energy[within_mask])
        irradiance_grid[unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] += energy_sum

        # # - Or Simply count the number of photons at each voxel and add the count to the irradiance grid
        # unique_indices, counts = torch.unique(within_indices, return_counts=True, dim=0)
        # irradiance_grid[unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] += counts.float()

        if num_show_images > 0 and cur_step in plot_step_indices:
            plotter.plot_wavefront_position(cur_pos.cpu().numpy(), cur_dir.cpu().numpy(), f"Step {cur_step} (Total: {within_indices.shape[0]})")

    return irradiance_grid.cpu().numpy()

# def run_monte_carlo_simulation(num_iterations: int, num_samplers_per_voxel: int = 8, pos_perturbation_scale: float = 0.45) -> np.ndarray:
#     avg_irradiance_grid = None    
#     for i in range(num_iterations):
#         initial_wavefront_pos, initial_wavefront_dir = generate_initial_wavefront(num_samplers_per_voxel, pos_perturbation_scale)
#         irradiance_grid, _ = simulate_wavefront_propagation(scene_ior, scene_ior_gradients, initial_wavefront_pos, initial_wavefront_dir, device, test_num_steps, test_delta_t, num_show_images=0)
#         print(f"Iteration {i+1}/{num_iterations} - Total number of points: {np.sum(irradiance_grid)}")
#         if avg_irradiance_grid is None:
#             avg_irradiance_grid = irradiance_grid
#         else:
#             avg_irradiance_grid = (avg_irradiance_grid * i + irradiance_grid) / (i + 1)
#     assert avg_irradiance_grid is not None, "The average irradiance grid should not be None"
#     return avg_irradiance_grid

# num_monte_carlo_iterations = 4
# avg_irradiance_grid = run_monte_carlo_simulation(num_monte_carlo_iterations, sampler_multiplier, pos_perturbation_scale)

def remove_under_floor(grid: np.ndarray, floor_height: int) -> np.ndarray:
    grid[:, :floor_height, :] = 0
    return grid
