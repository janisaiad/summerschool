# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# # Neural Operator Baseline for Heat Equation
# This notebook implements a Fourier Neural Operator (FNO) model using neuraloperator to solve the 2D heat equation.

# + [code]
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
from datetime import datetime
import logging
from neuralop.models import TFNO2d
from neuralop import Trainer
from torch.utils.data import TensorDataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU availability and configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    logger.warning("CUDA is not available, using CPU")

# + [code]
# Define parameters
nx = 30  # Number of points in x direction
ny = 30  # Number of points in y direction
nt = 500  # Number of time steps
alpha = 0.1  # Diffusion coefficient (matching training_fno_heat_big.py)

def solve_heat_square(initial_conditions, t, x, y, alpha=0.1):
    nx, ny = len(x), len(y)
    nt = len(t)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    solution = np.zeros((nx, ny, nt))
    solution[:, :, 0] = initial_conditions
    
    for n in range(1, nt):
        dt = t[n] - t[n-1]
        solution[:, :, n] = solution[:, :, n-1].copy()
        
        for i in range(nx):
            for j in range(ny):
                ip1 = (i + 1) % nx
                im1 = (i - 1) % nx
                jp1 = (j + 1) % ny
                jm1 = (j - 1) % ny
                
                laplacian = (solution[ip1, j, n-1] - 2*solution[i, j, n-1] + solution[im1, j, n-1]) / dx**2 + \
                           (solution[i, jp1, n-1] - 2*solution[i, j, n-1] + solution[i, jm1, n-1]) / dy**2
                
                solution[i, j, n] = solution[i, j, n-1] + dt * alpha * laplacian
    
    return solution

# Generate training data
def generate_data(n_samples=100):  # Reduced dataset size
    X, Y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))
    t = np.linspace(0,2,nt)
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    
    # Initialize data arrays
    inputs = np.zeros((n_samples, 1, nx, ny))
    outputs = np.zeros((n_samples, 1, nx, ny))
    
    # Generate half Gaussian and half sinusoidal data
    n_half = n_samples // 2
    
    # Gaussian part with tqdm
    means = np.random.uniform(0.3, 0.7, size=(n_half, 2))
    variances = np.random.uniform(0.01, 0.1, size=n_half)
    
    for i in tqdm(range(n_half), desc="Generating Gaussian data"):
        mean_x, mean_y = means[i]
        var = variances[i]
        init_cond = np.exp(-((X - mean_x)**2 + (Y - mean_y)**2)/(2*var))
        inputs[i, 0] = init_cond
        solution = solve_heat_square(init_cond, t, x, y, alpha)
        outputs[i, 0] = solution[:, :, -1]  # Take final time step
        
    # Sinusoidal part with tqdm
    freqs = np.linspace(0.5, 5, n_half)
    for i in tqdm(range(n_half), desc="Generating Sinusoidal data"):
        freq = freqs[i]
        init_cond = np.sin(2*np.pi*freq*X)*np.cos(2*np.pi*freq*Y) + \
                    np.cos(2*np.pi*freq*X)*np.sin(2*np.pi*freq*Y)
        idx = i + n_half
        inputs[idx, 0] = init_cond
        solution = solve_heat_square(init_cond, t, x, y, alpha)
        outputs[idx, 0] = solution[:, :, -1]  # Take final time step
    
    # Normalize data to prevent NaN losses
    inputs = (inputs - inputs.mean()) / (inputs.std() + 1e-8)
    outputs = (outputs - outputs.mean()) / (outputs.std() + 1e-8)
    
    return torch.FloatTensor(inputs).to(device), torch.FloatTensor(outputs).to(device)

# + [code]
# Create model with parameters matching training_fno_heat_big.py
model = TFNO2d(
    n_modes_width=12,  # Reduced from 30 to prevent instability
    n_modes_height=12,  # Reduced from 30 to prevent instability
    hidden_channels=32,
    in_channels=1,
    out_channels=1,
    projection_channels=64,
    n_layers=3,
    factorization='tucker',
    implementation='factorized'
).to(device)

# Generate smaller dataset
logger.info("Generating training data...")
train_x, train_y = generate_data(100)  # Reduced dataset size
logger.info("Generating test data...")
test_x, test_y = generate_data(20)  # Reduced dataset size

# Create DataLoaders
batch_size = 16  # Reduced batch size
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Configure trainer with custom training step and gradient clipping
class CustomTrainer:
    def __init__(self, model, n_epochs, device):
        self.model = model
        self.n_epochs = n_epochs
        self.device = device
        
    def train(self, train_loader, test_loaders, optimizer, scheduler):
        history = {'train_loss': [], 'test_loss': []}
        
        for epoch in tqdm(range(self.n_epochs), desc="Training"):
            # Training
            self.model.train()
            train_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                pred = self.model(x)
                loss = torch.nn.MSELoss()(pred, y)
                loss.backward()
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Testing
            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for x, y in test_loaders['test']:
                    pred = self.model(x)
                    loss = torch.nn.MSELoss()(pred, y)
                    test_loss += loss.item()
            
            test_loss /= len(test_loaders['test'])
            history['test_loss'].append(test_loss)
            
            if scheduler is not None:
                scheduler.step()
                
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{self.n_epochs}')
                logger.info(f'Train Loss: {train_loss:.6f}')
                logger.info(f'Test Loss: {test_loss:.6f}')
                
        return history

trainer = CustomTrainer(model=model, n_epochs=100, device=device)

# + [code]
# Train model
logger.info("Starting training...")

# Create test loader dictionary
test_loaders = {'test': test_loader}

# Define optimizer and scheduler with lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Reduced learning rate
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Train model
history = trainer.train(
    train_loader,
    test_loaders,
    optimizer=optimizer,
    scheduler=scheduler
)

# + [code]
# Save and plot results
save_path = "results/neuralop/"
os.makedirs(save_path, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.semilogy(history['train_loss'], label="Training loss")
plt.semilogy(history['test_loss'], label="Testing loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
plt.savefig(os.path.join(save_path, f"neural_operator_baseline_{timestamp}.png"))
plt.show()

# Save model
torch.save(model.state_dict(), os.path.join(save_path, f"model_{timestamp}.pt"))

logger.info(f"Training completed. Final train loss: {history['train_loss'][-1]:.6f}")
logger.info(f"Final test loss: {history['test_loss'][-1]:.6f}")
logger.info(f"Results saved to {save_path}")
# -
