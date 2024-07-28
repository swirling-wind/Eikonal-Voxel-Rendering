import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
from simulation.simulator import DEVICE
import os

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, hidden_layers: int, out_features: int, outermost_linear: bool=True,
                 first_omega_0: int=30, hidden_omega_0: float=30.0):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for _i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=int(hidden_omega_0)))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=int(hidden_omega_0)))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords) # type: ignore
        return output

def get_tensor_from_grid(voxel_grid: np.ndarray) -> torch.Tensor:
    transform = Compose([
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    voxel_tensor = transform(voxel_grid)
    assert isinstance(voxel_tensor, torch.Tensor), "Expected a tensor after transformation"
    return voxel_tensor

def get_coord_grid(sidelen: int, dim: int) -> torch.Tensor:
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1. '''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class VoxelFitting(Dataset):
    def __init__(self, voxel_grid: np.ndarray, sidelength: int):
        super().__init__()
        voxel_tensor = get_tensor_from_grid(voxel_grid)
        self.voxels = voxel_tensor.view(1, sidelength, sidelength, sidelength).permute(0, 2, 3, 1)
        self.coords = get_coord_grid(sidelength, dim=3)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError
        return self.coords, self.voxels

class ImageFitting(Dataset):
    def __init__(self, grid, sidelength: int):
        super().__init__()
        img = get_tensor_from_grid(grid)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_coord_grid(sidelength, dim=2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels
