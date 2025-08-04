import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os

def solve_heat_square(boundary_conditions:np.ndarray, initial_conditions:np.ndarray, t:np.ndarray, x:np.ndarray, y:np.ndarray, alpha:float=1.0)->np.ndarray:
    nx, ny = len(x), len(y)  # [scalar], [scalar]
    nt = len(t)  # [scalar]
    
    dx = x[1] - x[0]  # [scalar] 
    dy = y[1] - y[0]  # [scalar]
    
    solution = np.zeros((nx, ny, nt))  # [nx, ny, nt]
    solution[:, :, 0] = initial_conditions  # [nx, ny]
    
    for n in tqdm(range(1, nt), desc="Solving heat equation"):
        dt = t[n] - t[n-1]  # [scalar]
        solution[:, :, n] = solution[:, :, n-1].copy()  # [nx, ny]
        
        for i in range(nx):
            for j in range(ny):
                ip1 = (i + 1) % nx  # [scalar]
                im1 = (i - 1) % nx  # [scalar] 
                jp1 = (j + 1) % ny  # [scalar]
                jm1 = (j - 1) % ny  # [scalar]
                
                laplacian = (solution[ip1, j, n-1] - 2*solution[i, j, n-1] + solution[im1, j, n-1]) / dx**2 + \
                           (solution[i, jp1, n-1] - 2*solution[i, j, n-1] + solution[i, jm1, n-1]) / dy**2  # [scalar]
                
                solution[i, j, n] = solution[i, j, n-1] + dt * alpha * laplacian  # [scalar]
    
    return solution  # [nx, ny, nt]

def create_big_heat_data(N:int=1000, nx:int=30, ny:int=30, nt:int=500, alpha:float=0.05)->None:
    
    
    base_dir = "data/test_data/big_dataset_fno/heat2d"
    for subdir in ['mu', 'sol', 'xs']:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    N_half = N//2
    X, Y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))  # [nx, ny], [nx, ny]
    
    means = np.random.uniform(0.3, 0.7, size=(N_half, 2))  # [N_half, 2] for x,y means
    variances = np.random.uniform(0.01, 0.1, size=N_half)  # [N_half] for variance
    
    for i in tqdm(range(N_half), desc="Creating Gaussian heat data"):
        mean_x, mean_y = means[i]  # [scalar], [scalar]
        opposed_mean_x, opposed_mean_y = 1-mean_x, 1-mean_y  # [scalar], [scalar]
        var = variances[i]  # [scalar]
        random_sign = np.random.choice([-1, 1])
        initial_conditions = random_sign * np.exp(-((X - mean_x)**2 + (Y - mean_y)**2)/(2*var))+(1-random_sign) * np.exp(-((X - opposed_mean_x)**2 + (Y - opposed_mean_y)**2)/(2*var))   # [nx, ny] 
        
        T = 2
        t = np.linspace(0,T,nt)  # [nt]
        x = np.linspace(0,1,nx)  # [nx]
        y = np.linspace(0,1,ny)  # [ny]
        
        mu = np.expand_dims(initial_conditions, axis=-1)  # [nx, ny, 1]
        sol = solve_heat_square(None, initial_conditions, t, x, y, alpha)  # [nx, ny, nt]
        xs = np.stack([X, Y], axis=-1)  # [nx, ny, 2]
        
        np.save(f"{base_dir}/mu/mu_{i}.npy", mu)
        np.save(f"{base_dir}/sol/sol_{i}.npy", sol)
        np.save(f"{base_dir}/xs/xs_{i}.npy", xs)

    freqs = np.linspace(0.5, 5, N_half)  # [N_half]
    
    for i, freq in tqdm(enumerate(freqs), desc="Creating sinusoidal heat data"):
        initial_conditions = np.sin(2*np.pi*freq*X)*np.cos(2*np.pi*freq*Y) + \
                           np.cos(2*np.pi*freq*X)*np.sin(2*np.pi*freq*Y)  # [nx, ny]
        
        t = np.linspace(0,1,nt)  # [nt]
        x = np.linspace(0,1,nx)  # [nx]
        y = np.linspace(0,1,ny)  # [ny]
        
        mu = np.expand_dims(initial_conditions, axis=-1)  # [nx, ny, 1]
        sol = solve_heat_square(None, initial_conditions, t, x, y, alpha)  # [nx, ny, nt]
        xs = np.stack([X, Y], axis=-1)  # [nx, ny, 2]
        
        idx = i + N_half
        np.save(f"{base_dir}/mu/mu_{idx}.npy", mu)
        np.save(f"{base_dir}/sol/sol_{idx}.npy", sol)
        np.save(f"{base_dir}/xs/xs_{idx}.npy", xs)

    params = {
        "n_mu": N,
        "nx": nx, 
        "ny": ny,
        "nt": nt,
        "alpha": alpha,
        "freq_range": [float(freqs[0]), float(freqs[-1])],
        "gaussian_means_range": [float(means.min()), float(means.max())],
        "gaussian_variance_range": [float(variances.min()), float(variances.max())],
        "initial_conditions": ["Gaussian: exp(-((x-mean_x)^2 + (y-mean_y)^2)/(2*var))", 
                             "Sinusoidal: sin(2πfx)cos(2πfy) + cos(2πfx)sin(2πfy)"]
    }
    with open(f"{base_dir}/params.json", "w") as f:
        json.dump(params, f)

if __name__ == "__main__":
    create_big_heat_data()


# stability condition for heat equation: dt <= dx^2 / (2*alpha) which is 