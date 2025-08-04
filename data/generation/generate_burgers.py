import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os

def solve_burgers_square(initial_conditions:np.ndarray, t:np.ndarray, x:np.ndarray, y:np.ndarray)->np.ndarray:
    '''
    Solve 2D linear Burgers equation with periodic BC using upwind scheme
    # Format: (2,nt,nx,ny) for (u,v) components
    # Equations:
    # u_t + u_x + v_y = 0
    # v_t + u_x + v_y = 0
    '''
    nx, ny = len(x), len(y)
    nt = len(t)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    u = np.zeros((nt, nx, ny))
    v = np.zeros((nt, nx, ny))
    
    u[0] = initial_conditions[0]
    v[0] = initial_conditions[1]
    
    for n in tqdm(range(1, nt), desc="Solving linear Burgers equation"):
        dt = t[n] - t[n-1]
        
        # CFL condition
        dt = min(dt, 0.5*min(dx,dy))
        
        for i in range(nx):
            for j in range(ny):
                # Periodic indices
                ip1 = (i + 1) % nx
                im1 = (i - 1) % nx
                jp1 = (j + 1) % ny
                jm1 = (j - 1) % ny
                
                # Upwind differences for u
                flux_x_u = (u[n-1,i,j] - u[n-1,im1,j])/dx
                flux_y_u = (u[n-1,i,j] - u[n-1,i,jm1])/dy
                
                # Upwind differences for v  
                flux_x_v = (v[n-1,i,j] - v[n-1,im1,j])/dx
                flux_y_v = (v[n-1,i,j] - v[n-1,i,jm1])/dy
                
                # Update
                u[n,i,j] = u[n-1,i,j] - dt * (flux_x_u + flux_y_u)
                v[n,i,j] = v[n-1,i,j] - dt * (flux_x_v + flux_y_v)
    
    return np.stack([u, v])

def create_burgers_data(n_mu:int, nt:int, nx:int, ny:int)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Generate data for different initial conditions with Gaussian waveform"""
    os.makedirs("data/test_data/example_data/burgers2d", exist_ok=True)
    
    for i in tqdm(range(n_mu), desc="Creating linear Burgers data"):
        # Random amplitudes between -1 and 1
        u_amp = np.random.uniform(-1, 1)
        v_amp = np.random.uniform(-1, 1)
        
        x = np.linspace(0, 2*np.pi, nx)
        y = np.linspace(0, 2*np.pi, ny)
        X, Y = np.meshgrid(x, y)
        
        # Create 2D Gaussian centered in the domain
        x0, y0 = np.pi, np.pi  # Center of the domain
        sigma = 3  # Width of Gaussian
        gaussian = np.exp(-((X-x0)**2 + (Y-y0)**2)/(2*sigma**2))
        
        # Initial conditions: Gaussian shape
        u0 = gaussian * u_amp
        v0 = gaussian * v_amp
        initial_conditions = np.array([u0, v0])
        
        t = np.linspace(0, 1, nt)
        sol = solve_burgers_square(initial_conditions, t, x, y)
        
        X, Y, T = np.meshgrid(x, y, t)
        points = np.column_stack((X.ravel(), Y.ravel(), T.ravel()))
        
        np.save(f"data/test_data/example_data/burgers2d/mu/mu_{i}.npy", initial_conditions)
        np.save(f"data/test_data/example_data/burgers2d/xs/xs_{i}.npy", points)
        np.save(f"data/test_data/example_data/burgers2d/sol/sol_{i}.npy", sol)
        
        with open(f"data/test_data/example_data/burgers2d/params/params_{i}.json", "w") as f:
            json.dump({
                "u_amplitude": float(u_amp),
                "v_amplitude": float(v_amp), 
                "sigma": float(sigma),
                "nt": nt,
                "nx": nx,
                "ny": ny
            }, f)

if __name__ == "__main__":
    create_burgers_data(n_mu=40, nt=20, nx=15, ny=15)
