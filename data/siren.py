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


def get_mgrid(sidelen: int|tuple, dim: int) -> torch.Tensor:
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1. '''

    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32) # type: ignore
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32) # type: ignore
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

class VoxelFitting(Dataset):
    def __init__(self, voxel_grid: np.ndarray, sidelength: int|tuple):
        super().__init__()
        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength, sidelength)

        self.coords = get_mgrid(sidelength, dim=3).reshape(-1, 3)

        self.transform = Compose([
            ToTensor(),
        ])
        irrad_grid = self.transform(voxel_grid)
        self.voxels = irrad_grid.view(1, *sidelength).permute(0, 2, 3, 1).reshape(-1) # type: ignore

    # def __len__(self):
    #     return 1

    # def __getitem__(self, idx):
    #     if idx > 0: raise IndexError
    #     return self.coords, self.voxels
    
    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.voxels[idx]


class ImageFitting(Dataset):
    def __init__(self, dataset, sidelength: int|tuple):
        super().__init__()
        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)

        self.coords = get_mgrid(sidelength, dim=2)

        self.transform = Compose([
            Resize(sidelength),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
        self.dataset = dataset
        img = self.transform(self.dataset)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        
    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        
        return self.coords, self.pixels