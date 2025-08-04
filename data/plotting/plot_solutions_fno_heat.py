import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

type_of_problem = "heat_fno"

def create_heat_animation(sol:np.ndarray, x:np.ndarray, y:np.ndarray, nt:int, index:int):
    """Create animation of heat equation solution over time"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)  # [nx, ny], [nx, ny]
    
    surf = ax.plot_surface(X, Y, sol[0].T, cmap=cm.viridis, alpha=0.8)  # [nx, ny]
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    
    def update(frame):
        ax.clear()
        surf = ax.plot_surface(X, Y, sol[frame].T, cmap=cm.viridis, alpha=0.8)  # [nx, ny]
        ax.set_xlabel('x')
        ax.set_ylabel('y') 
        ax.set_zlabel('Temperature')
        ax.set_title(f'Heat distribution at t={frame}')
        return surf,

    anim = FuncAnimation(fig, update, frames=nt, interval=200, blit=False)
    
    os.makedirs(f"data/animation/{type_of_problem}", exist_ok=True)
    
    writer = PillowWriter(fps=5)
    anim.save(f"data/animation/{type_of_problem}/heat_animation_{index}.gif", writer=writer)
    plt.close()

def plot_heat_solution(sol:np.ndarray, x:np.ndarray, y:np.ndarray, t_interior:int, index:int):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)  # [nx, ny], [nx, ny]
    surf = ax.plot_surface(X, Y, sol[t_interior].T, cmap=cm.viridis, alpha=0.8)  # [nx, ny]
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    ax.set_title(f'Heat distribution at t={t_interior}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    os.makedirs(f"data/plots/{type_of_problem}", exist_ok=True)
    plt.savefig(f"data/plots/{type_of_problem}/heat_solution_{index}.png")
    plt.close()

if __name__ == "__main__":
    # Load the full dataset
    sol = np.load("data/test_data/example_data_fno/heat2d/sol.npy")  # [N, nx, ny, nt]
    mu = np.load("data/test_data/example_data_fno/heat2d/mu.npy")  # [N, nx, ny, 1]
    
    with open("data/test_data/example_data_fno/heat2d/params.json", "r") as f:
        params = json.load(f)
    
    nx = params["nx"]  # [scalar]
    ny = params["ny"]  # [scalar]
    nt = params["nt"]  # [scalar]
    
    x = np.linspace(0, 1, nx)  # [nx]
    y = np.linspace(0, 1, ny)  # [ny]
    
    # Plot each solution and create animation
    for index in tqdm(range(params["n_mu"]), desc="Creating plots and animations"):
        plot_heat_solution(sol[index], x, y, nt-1, index)  # Plot final time
        create_heat_animation(sol[index], x, y, nt, index)