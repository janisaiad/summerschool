import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import json

type_of_problem = "burgers2d"

def plot_heat_solution(sol:np.ndarray, x:np.ndarray, y:np.ndarray, t_interior:float, index:int):
    """
    Plot the solution of the Burgers equation
    
    Parameters:
    - sol: solution array of shape (2, nt, nx, ny) where 2 represents (u,v) components
    - x, y: spatial coordinates
    - t_interior: time index to plot
    - index: index for saving the file
    """
    fig = plt.figure(figsize=(12, 10))
    
    # Plot u component
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf1 = ax1.plot_surface(X, Y, sol[0, t_interior].T, cmap=cm.viridis, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    ax1.set_title(f'u component at t={t_interior}')
    fig.colorbar(surf1, shrink=0.5, aspect=5)
    
    # Plot v component
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, sol[1, t_interior].T, cmap=cm.viridis, alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('v')
    ax2.set_title(f'v component at t={t_interior}')
    fig.colorbar(surf2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(f"data/plots/{type_of_problem}/burgers_solution_{index}.png")
    plt.close()

if __name__ == "__main__":
    for index in range(40):
        sol = np.load(f"data/test_data/example_data/{type_of_problem}/sol/sol_{index}.npy")
        matrix = np.load(f"data/test_data/example_data/{type_of_problem}/mu/mu_{index}.npy")
        xs = np.load(f"data/test_data/example_data/{type_of_problem}/xs/xs_{index}.npy")
        X = np.unique(xs[:,0])
        Y = np.unique(xs[:,1])
        
        with open(f"data/test_data/example_data/{type_of_problem}/params/params_{index}.json", "r") as f:
            params = json.load(f)
        nt = params["nt"]
        
        plot_heat_solution(sol, X, Y, nt//2 - 1, index)