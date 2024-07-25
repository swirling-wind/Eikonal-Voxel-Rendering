import torch
import torch.nn as nn
import numpy as np
from simulation.simulator import DEVICE

class IrradianceNet(nn.Module):
    def __init__(self):
        super(IrradianceNet, self).__init__()
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

def prepare_data(irradiance: np.ndarray, floor_height: int, val_ratio=0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    irradiance_tensor = torch.tensor(irradiance[:, floor_height:, :], dtype=torch.float32)
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
    train_size = int((1- val_ratio) * len(inputs))
    train_inputs, val_inputs = inputs[:], inputs[train_size:] 
    train_targets, val_targets = targets[:], targets[train_size:]

    return train_inputs, val_inputs, train_targets, val_targets

def train_model(model: IrradianceNet, train_inputs: torch.Tensor, val_inputs: torch.Tensor, 
                train_targets: torch.Tensor, val_targets: torch.Tensor, 
                num_epochs=100, batch_size=1024):
    torch.cuda.empty_cache()
    print("Device:", DEVICE)
    
    prev_lr = 0.001

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=prev_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(train_inputs), batch_size):
            batch_inputs = train_inputs[i:i+batch_size]
            batch_targets = train_targets[i:i+batch_size]
            
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

            if scheduler.get_last_lr()[0] != prev_lr:
                prev_lr = scheduler.get_last_lr()[0]
                print("Current learning rate:", prev_lr)

    torch.cuda.empty_cache()
    return model

def generate_irradiance_field_3d(model: IrradianceNet, floor_height: int, pad: bool=False, batch_size: int = 2048) -> torch.Tensor:
    NUM_X, NUM_Y, NUM_Z = 128, 128, 128
    x = np.arange(NUM_X)
    z = np.arange(NUM_Z)
    X, Z = np.meshgrid(x, z)
    
    predictions = []
    with torch.no_grad():
        for y in range(NUM_Y - floor_height):
            coords = np.stack((X, np.full_like(X, y), Z), axis=-1).reshape(-1, 3)
            
            y_predictions = []
            for i in range(0, len(coords), batch_size):
                batch_coords = coords[i:i+batch_size]
                inputs = torch.tensor(batch_coords, dtype=torch.float32, device=DEVICE)
                batch_predictions = model(inputs)
                y_predictions.append(batch_predictions)
            
            y_predictions = torch.cat(y_predictions)
            y_predictions = y_predictions.reshape(NUM_X, NUM_Z)
            predictions.append(y_predictions)
    
    predictions = torch.stack(predictions)
    predictions = predictions.transpose(0, 1)
    
    # 填充低于floor_height的区域
    if pad:
        padding = (0, 0, floor_height, 0, 0, 0) 
        predictions = torch.nn.functional.pad(predictions, padding, mode='constant', value=0)
    
    return predictions