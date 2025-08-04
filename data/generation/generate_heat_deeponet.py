import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os

def solve_heat_square(boundary_conditions:np.ndarray, initial_conditions:np.ndarray, t:np.ndarray, x:np.ndarray, y:np.ndarray, alpha:float=1.0)->np.ndarray:
    nx, ny = len(x), len(y)
    nt = len(t)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    solution = np.zeros((nt, nx, ny))
    
    solution[0, :, :] = initial_conditions
    
    solution[0, 0, :] = boundary_conditions[0]
    solution[0, :, 0] = boundary_conditions[1]
    solution[0, -1, :] = boundary_conditions[2]
    solution[0, :, -1] = boundary_conditions[3]
    
    for n in tqdm(range(1, nt), desc="Solving heat equation"):
        dt = t[n] - t[n-1]
        
        solution[n] = solution[n-1].copy()
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                laplacian = (solution[n-1, i+1, j] - 2*solution[n-1, i, j] + solution[n-1, i-1, j]) / dx**2 + \
                           (solution[n-1, i, j+1] - 2*solution[n-1, i, j] + solution[n-1, i, j-1]) / dy**2
                
                solution[n, i, j] = solution[n-1, i, j] + dt * alpha * laplacian
        
        solution[n, 0, :] = boundary_conditions[0]
        solution[n, :, 0] = boundary_conditions[1]
        solution[n, -1, :] = boundary_conditions[2]
        solution[n, :, -1] = boundary_conditions[3]
    
    return solution

def create_heat_data(n_mu:int,nt:int,nx:int,ny:int,alpha:float=0.05)->None:
    base_dir = "data/test_data/big_dataset_deeponet/heat2d"
    for subdir in ['mu', 'sol', 'xs']:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
        
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    t = np.linspace(0,1,nt)
    
    cheb_x = np.cos(np.pi * (2*np.arange(8) + 1) / (2*8))
    cheb_y = np.cos(np.pi * (2*np.arange(8) + 1) / (2*8))
    X_sens, Y_sens = np.meshgrid(cheb_x, cheb_y)
    
    # Create time points for each spatial point
    T_sens = np.linspace(0, 1, nt)
    X_sens_t, Y_sens_t, T_mesh = np.meshgrid(cheb_x, cheb_y, T_sens)
    sensor_points = np.column_stack((X_sens_t.ravel(), Y_sens_t.ravel(), T_mesh.ravel()))
    
    for i in tqdm(range(n_mu)):
        coeffs = np.random.uniform(-1, 1, (8, 8))
        X, Y = np.meshgrid(x, y)
        initial_condition = np.zeros_like(X)
        
        for m in range(5):
            for n in range(5):
                initial_condition += coeffs[m,n] * np.cos(m*np.pi*X) * np.cos(n*np.pi*Y)
                
        boundary_conditions = [
            initial_condition[0,:],
            initial_condition[:,0],
            initial_condition[-1,:],
            initial_condition[:,-1]
        ]
        
        sol = solve_heat_square(boundary_conditions, initial_condition, t, x, y, alpha)
        
        np.save(f"{base_dir}/mu/mu_{i}.npy", initial_condition)
        np.save(f"{base_dir}/xs/xs_{i}.npy", sensor_points)
        np.save(f"{base_dir}/sol/sol_{i}.npy", sol)
        
    params = {
        "n_mu": n_mu,
        "nt": nt,
        "nx": nx,
        "ny": ny,
        "alpha": alpha,
        "n_cheb_points": 8
    }
    with open(f"{base_dir}/params.json", "w") as f:
        json.dump(params, f)

if __name__ == "__main__":
    create_heat_data(n_mu=32, nt=40, nx=8, ny=8, alpha=0.05)