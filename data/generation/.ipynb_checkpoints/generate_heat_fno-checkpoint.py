import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

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

def create_heat_data(N:int, nx:int, ny:int, nt:int, alpha:float=1.0)->tuple[np.ndarray, np.ndarray]:
    freqs = np.linspace(0.5, 1, N)  # [N]
    mu = np.zeros((N, nx, ny, 1))  # [N, nx, ny, 1]
    sol = np.zeros((N, nx, ny, nt))  # [N, nx, ny, nt]
    xs = np.zeros((N, nx, ny, 2))  # [N, nx, ny, 2]
    
    for i, freq in tqdm(enumerate(freqs), desc="Creating heat data"):
        X, Y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))  # [nx, ny], [nx, ny]
        initial_conditions = np.sin(2*np.pi*freq*X)*np.cos(2*np.pi*freq*Y) + \
                           np.cos(2*np.pi*freq*X)*np.sin(2*np.pi*freq*Y)  # [nx, ny]
        
        t = np.linspace(0,1,nt)  # [nt]
        x = np.linspace(0,1,nx)  # [nx]
        y = np.linspace(0,1,ny)  # [ny]
        
        mu[i, :, :, 0] = initial_conditions  # [nx, ny]
        sol[i] = solve_heat_square(None, initial_conditions, t, x, y, alpha)  # [nx, ny, nt]
        xs[i, :, :, :] = np.stack([X, Y], axis=-1)  # [nx, ny, 2]
    np.save("data/test_data/example_data_fno/heat2d/mu.npy", mu)  # [N, nx, ny, 1]
    np.save("data/test_data/example_data_fno/heat2d/sol.npy", sol)  # [N, nx, ny, nt]
    np.save("data/test_data/example_data_fno/heat2d/xs.npy", xs)  # [N, nx, ny, 2]
    params = {
        "n_mu": N,
        "nx": nx, 
        "ny": ny,
        "nt": nt,
        "alpha": alpha,
        "freq_range": [float(freqs[0]), float(freqs[-1])],
        "initial_conditions": "np.sin(2*np.pi*freq*X)*np.cos(2*np.pi*freq*Y) + np.cos(2*np.pi*freq*X)*np.sin(2*np.pi*freq*Y)"
    }
    with open("data/test_data/example_data_fno/heat2d/params.json", "w") as f:
        json.dump(params, f)

if __name__ == "__main__":
    create_heat_data(N=40, nx=20, ny=20, nt=20, alpha=0.01)