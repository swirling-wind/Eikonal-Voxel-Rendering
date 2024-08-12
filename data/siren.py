import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
from simulation.simulator import DEVICE, normalize_by_max
from setup.scene_utils import get_floor_height
import os

class SineLayer(nn.Module):
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
    
class SirenFitter:
    def __init__(self, input_arr: np.ndarray, scene_config: dict,
                 hidden_features: int = 256, hidden_layers: int = 3, omega: int = 20):
        self.scene_name = scene_config["Name"]
        self.floor_height = get_floor_height(scene_config["NUM XYZ"][1], scene_config["Floor Ratio"])
        self.sample_multiplier = scene_config["Sampler Num"]
        self.input_arr = input_arr[:, self.floor_height:, :]
        self.cropped_shape = self.input_arr.shape
        self.omega = omega
        self.siren = Siren(in_features=3, out_features=1, 
                           hidden_features=hidden_features, hidden_layers=hidden_layers, 
                           outermost_linear=True,
                           first_omega_0=omega, hidden_omega_0=omega)
    
    def fit(self, total_epochs: int = 20, batch_size: int = 20000, lr: float = 5e-4, load_file: bool = True, patience: int = 3, show_epoch_interval: int = 5):
        model_path = os.path.join(os.getcwd(), "data", "saves", f"SIREN({self.scene_name})({self.sample_multiplier}-samplers)({total_epochs}-epoches)({self.omega}-omega).pt")
        if load_file == False or (not os.path.exists(model_path)):     
            if not os.path.exists(model_path):
                print("[ Not found ] SIREN model file \"{}\" does not exist. Start training the model...".format(model_path.split("\\")[-1]))
            self.train(total_epochs, batch_size, lr, patience, show_epoch_interval)
            torch.save(self.siren.state_dict(), model_path)
        else:
            self.siren.load_state_dict(torch.load(model_path))
            self.siren.to(DEVICE)
            print("[ Loaded ] SIREN model from \"{}\"".format(model_path.split("\\")[-1]))
    
    def train(self, total_epochs: int = 20, batch_size: int = 20000, lr: float = 5e-4, patience: int = 5, show_epoch_interval: int = 5):           
        torch.cuda.empty_cache()
        irrad_data = VoxelFitting(self.input_arr, sidelength=self.input_arr.shape)
        dataloader = DataLoader(irrad_data, batch_size, sampler=RandomSampler(irrad_data),
                                pin_memory=True)

        self.siren.to(DEVICE)
        optim = torch.optim.Adam(lr=lr, params=self.siren.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.2, patience=patience)

        self.siren.train()
        for cur_epoch in range(total_epochs):
            epoch_loss = 0
            for model_input, ground_truth in dataloader:
                model_input, ground_truth = model_input.to(DEVICE), ground_truth.to(DEVICE)
                model_output = self.siren(model_input)
                model_output = model_output.view(ground_truth.shape)

                loss = ((model_output - ground_truth) ** 2).mean()

                # epsilon = 1e-4  # Add to log to prevent NaN
                # - Use log loss [NOT WORK. INFER RESULT IS LIKE HEAT WAVE]                
                # loss = torch.log((model_output - ground_truth) ** 2 + epsilon).mean()

                # - Use log loss with abs [NOT WORK. THE CONVERGENCE IS TOO SLOW AND ONLY CAPTURES NOISE]
                # loss = torch.log(torch.abs(model_output - ground_truth) + epsilon).mean()

                # - Use relative error as loss [NOT WORK. INFER RESULT IS CONSTANT 0 VALUE]
                # relative_error = (model_output - ground_truth) / (ground_truth + epsilon)
                # loss = (relative_error ** 2).mean()

                epoch_loss += loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()

            epoch_loss /= len(dataloader)
            scheduler.step(epoch_loss)

            if cur_epoch == total_epochs - 1 or cur_epoch == 0 or (cur_epoch+1) % show_epoch_interval == 0:
                print("Epoch: {} \tAverage Loss: {:.4f} \tLearning Rate: {}".format(cur_epoch + 1, epoch_loss,
                                                                                    scheduler.get_last_lr()))

    def infer(self, pad: bool = True) -> np.ndarray:
        torch.cuda.empty_cache()
        infer_coord = get_mgrid(self.cropped_shape, dim=3).to(DEVICE)
        self.siren.eval()
        with torch.no_grad():
            infer_output: torch.Tensor = self.siren(infer_coord).view(self.cropped_shape).detach()
        
        if pad:
            padding = (0, 0, self.floor_height, 0, 0, 0) 
            infer_output = torch.nn.functional.pad(infer_output, padding, mode='constant', value=0)
        
        # Normalize the output to 0-255
        # infer_output = (infer_output - infer_output.min()) / infer_output.max() * 255
        
        return infer_output.cpu().numpy()

def siren_post_process(siren_res: np.ndarray, gamma = 0.6) -> np.ndarray:
    normalized_siren_res = normalize_by_max(siren_res)
    corrected_siren_res = ((normalized_siren_res / 255.0) ** (1.0 / gamma)) * 255.0
    return corrected_siren_res

