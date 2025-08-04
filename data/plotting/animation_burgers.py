import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

type_of_problem = "burgers2d"

def create_burgers_animation(sol:np.ndarray, x:np.ndarray, y:np.ndarray, nt:int, index:int):
    """sol: (2,nt,nx,ny) array for (u,v) components"""
    fig = plt.figure(figsize=(12, 10))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(x, y)
    
    surf1 = ax1.plot_surface(X, Y, sol[0, 0].T, cmap=cm.viridis, alpha=0.8)
    surf2 = ax2.plot_surface(X, Y, sol[1, 0].T, cmap=cm.viridis, alpha=0.8)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('v')
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        surf1 = ax1.plot_surface(X, Y, sol[0, frame].T, cmap=cm.viridis, alpha=0.8)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u')
        ax1.set_title(f'u component at t={frame}')
        
        surf2 = ax2.plot_surface(X, Y, sol[1, frame].T, cmap=cm.viridis, alpha=0.8)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('v')
        ax2.set_title(f'v component at t={frame}')
        
        return surf1, surf2

    anim = FuncAnimation(fig, update, frames=nt, interval=200, blit=False)
    
    os.makedirs(f"data/animation/{type_of_problem}", exist_ok=True)
    
    writer = PillowWriter(fps=5)
    anim.save(f"data/animation/{type_of_problem}/burgers_animation_{index}.gif", writer=writer)
    plt.close()

if __name__ == "__main__":
    for index in tqdm(range(40), desc="Creating animations"):
        sol = np.load(f"data/test_data/example_data/{type_of_problem}/sol_{index}.npy")
        matrix = np.load(f"data/test_data/example_data/{type_of_problem}/mu_{index}.npy")
        xs = np.load(f"data/test_data/example_data/{type_of_problem}/xs_{index}.npy")
        X = np.unique(xs[:,0])
        Y = np.unique(xs[:,1])
        
        with open(f"data/test_data/example_data/{type_of_problem}/params.json", "r") as f:
            params = json.load(f)
        nt = params["nt"]
        
        create_burgers_animation(sol, X, Y, nt, index)