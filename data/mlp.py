import torch
import torch.nn as nn
import numpy as np
from simulation.simulator import DEVICE
from simulation.simulate_utils import normalize_by_max
from setup.scene_utils import get_floor_height
import os

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
    def forward(self, x):
        return self.model(x).squeeze()
    
class MLPFitter:
    def __init__(self, irradiance: np.ndarray, scene_config: dict,
                 num_epoches=500, batch_size=20000, load_model_file=True, val_ratio=0.1):
        self.irradiance = irradiance

        self.scene_name = scene_config["Name"]
        self.floor_height = get_floor_height(scene_config["Num XYZ"][1], scene_config["Floor Ratio"])
        self.num_xyz = scene_config["Num XYZ"]
        self.sampler_multiplier = scene_config["Sampler Num"]

        self.val_ratio = val_ratio
        self.train_inputs, self.val_inputs, self.train_targets, self.val_targets = self._prepare_data()
        self.model = MLP().to(DEVICE)

        model_path = os.path.join(os.getcwd(), "data", "mlp_saves", f"MLP({self.scene_name})({self.sampler_multiplier}-samplers)({num_epoches}-epoches).pt")
        if load_model_file == False or (not os.path.exists(model_path)):
            if not os.path.exists(model_path):
                print("[ Not found ] model file \"{}\" does not exist. Start training the model...".format(model_path.split("\\")[-1]))
            self.train(num_epoches, batch_size)
            torch.save(self.model.state_dict(), model_path)
        else:
            self.model.load_state_dict(torch.load(model_path))
            print("[ Loaded ] model from \"{}\"".format(model_path.split("\\")[-1]))

    def _prepare_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        irradiance_tensor = torch.tensor(self.irradiance[:, self.floor_height:, :], dtype=torch.float32)
        x = np.arange(irradiance_tensor.shape[0])
        z = np.arange(irradiance_tensor.shape[2])
        X, Z = np.meshgrid(x, z)    
        inputs = []
        targets = []
        for y in range(irradiance_tensor.shape[1]):
            coords = np.stack((X, np.full_like(X, y), Z), axis=-1).reshape(-1, 3)
            inputs.append(coords)
            targets.append(irradiance_tensor[:, y, :].flatten())
        
        inputs = np.concatenate(inputs)
        targets = torch.cat(targets)

        print("Inputs shape:", inputs.shape, "; Targets shape:", targets.shape)
        inputs = torch.from_numpy(inputs).float().to(DEVICE)
        targets = targets.clone().detach().to(DEVICE)

        # Random shuffle
        SEED = 42 # For reproducibility
        torch.manual_seed(SEED)
        indices = torch.randperm(len(inputs))
        inputs = inputs[indices]
        targets = targets[indices]

        # train_inputs use all inputs because we use all data for training to get overfitting
        train_size = int((1- self.val_ratio) * len(inputs))
        train_inputs, val_inputs = inputs[:], inputs[train_size:] 
        train_targets, val_targets = targets[:], targets[train_size:]

        return train_inputs, val_inputs, train_targets, val_targets

    def train(self, num_epochs, batch_size):
        torch.cuda.empty_cache()
        print("Device:", DEVICE)
        
        prev_lr = 5e-4

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=prev_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        
        for epoch in range(num_epochs):
            self.model.train()
            for i in range(0, len(self.train_inputs), batch_size):
                batch_inputs = self.train_inputs[i:i+batch_size]
                batch_targets = self.train_targets[i:i+batch_size]
                
                outputs = self.model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.val_inputs)
                val_loss = criterion(val_outputs, self.val_targets)
                scheduler.step(val_loss)

            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

                if scheduler.get_last_lr()[0] != prev_lr:
                    prev_lr = scheduler.get_last_lr()[0]
                    print("Current learning rate:", prev_lr)
        torch.cuda.empty_cache()

    def predict(self, pad: bool=False, batch_size: int = 4096) -> np.ndarray:
        NUM_X, NUM_Y, NUM_Z = self.num_xyz
        x = np.arange(NUM_X)
        z = np.arange(NUM_Z)
        X, Z = np.meshgrid(x, z)
        
        predictions = []

        self.model.eval()
        with torch.no_grad():
            for y in range(NUM_Y - self.floor_height):
                coords = np.stack((X, np.full_like(X, y), Z), axis=-1).reshape(-1, 3)
                
                y_predictions = []
                for i in range(0, len(coords), batch_size):
                    batch_coords = coords[i:i+batch_size]
                    inputs = torch.tensor(batch_coords, dtype=torch.float32, device=DEVICE)
                    batch_predictions = self.model(inputs)
                    y_predictions.append(batch_predictions)
                
                y_predictions = torch.cat(y_predictions)
                y_predictions = y_predictions.reshape(NUM_X, NUM_Z)
                predictions.append(y_predictions)
        
        predictions = torch.stack(predictions)
        predictions = predictions.transpose(0, 1)
        
        # Pad the predictions to match the original irradiance field shape
        if pad:
            padding = (0, 0, self.floor_height, 0, 0, 0) 
            predictions = torch.nn.functional.pad(predictions, padding, mode='constant', value=0)
        
        return predictions.cpu().numpy()

def mlp_post_process(mlp_res: np.ndarray, gamma: float | None) -> np.ndarray:
    normalized_mlp_res = normalize_by_max(mlp_res)
    if gamma is not None:
        corrected_mlp_res = ((normalized_mlp_res / 255.0) ** (1.0 / gamma)) * 255.0
        return corrected_mlp_res
    
    return normalized_mlp_res