import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

type_of_problem = "heat_fno_big"

def create_heat_animation(sol:np.ndarray, xs:np.ndarray, nt:int, index:int, is_gaussian:bool):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = xs[..., 0], xs[..., 1]  # [nx, ny], [nx, ny]
    
    surf = ax.plot_surface(X, Y, sol[..., 0], cmap=cm.viridis, alpha=0.8)  # [nx, ny]
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    
    def update(frame):
        ax.clear()
        surf = ax.plot_surface(X, Y, sol[..., frame], cmap=cm.viridis, alpha=0.8)  # [nx, ny]
        ax.set_xlabel('x')
        ax.set_ylabel('y') 
        ax.set_zlabel('Temperature')
        title = f'Heat distribution at t={frame} ({"Gaussian" if is_gaussian else "Sinusoidal"} IC)'
        ax.set_title(title)
        return surf,

    anim = FuncAnimation(fig, update, frames=nt, interval=200, blit=False)
    
    os.makedirs(f"data/animation/{type_of_problem}", exist_ok=True)
    
    writer = PillowWriter(fps=5)
    ic_type = "gaussian" if is_gaussian else "sinusoidal"
    anim.save(f"data/animation/{type_of_problem}/heat_animation_{ic_type}_{index}.gif", writer=writer)
    plt.close()

def plot_heat_solution(sol:np.ndarray, xs:np.ndarray, t_interior:int, index:int, is_gaussian:bool):
    print(sol.shape)
    print(xs.shape)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = xs[..., 0], xs[..., 1]  # [nx, ny], [nx, ny]
    surf = ax.plot_surface(X, Y, sol[..., t_interior], cmap=cm.viridis, alpha=0.8)  # [nx, ny]
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    title = f'Heat distribution at t={t_interior} ({"Gaussian" if is_gaussian else "Sinusoidal"} IC)'
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    os.makedirs(f"data/plots/{type_of_problem}", exist_ok=True)
    ic_type = "gaussian" if is_gaussian else "sinusoidal"
    plt.savefig(f"data/plots/{type_of_problem}/heat_solution_{ic_type}_{index}.png")
    plt.close()

if __name__ == "__main__":
    
    with open("data/test_data/big_dataset_fno/heat2d/params.json", "r") as f:
        params = json.load(f)
    
    nx = params["nx"]  # [scalar]
    ny = params["ny"]  # [scalar]
    nt = params["nt"]  # [scalar]
    N_half = params["n_mu"] // 2  # [scalar]

    for index in tqdm(range(params["n_mu"]), desc="Creating plots and animations"):
        is_gaussian = index < N_half
        sol = np.load(f"data/test_data/big_dataset_fno/heat2d/sol/sol_{index}.npy")  # [N, nx, ny, nt]
        mu = np.load(f"data/test_data/big_dataset_fno/heat2d/mu/mu_{index}.npy")  # [N, nx, ny, 1]
        xs = np.load(f"data/test_data/big_dataset_fno/heat2d/xs/xs_{index}.npy")  # [N, nx, ny, 2]
        plot_heat_solution(sol, xs, nt-1, index % N_half, is_gaussian)  # Plot final time
        create_heat_animation(sol, xs, nt, index % N_half, is_gaussian)