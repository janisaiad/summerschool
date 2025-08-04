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

# + [markdown]
# # Deep XDE Baseline for Heat Equation
# This notebook implements a PINN model using DeepXDE to solve the 2D heat equation.

# + [code]
import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU availability and configure device
device = "/CPU:0"  # Default to CPU
if tf.test.is_built_with_cuda():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            device = "/GPU:0"
            logger.info(f"Using GPU: {physical_devices[0]}")
        except RuntimeError as e:
            logger.warning(f"Unable to use GPU: {e}")
            logger.warning("Falling back to CPU")
    else:
        logger.warning("No GPU devices found, using CPU")
else:
    logger.warning("CUDA is not available, using CPU")

# + [code]
# Define parameters
nx = 30  # Number of points in x direction
ny = 30  # Number of points in y direction
alpha = 0.05  # Diffusion coefficient
t1 = 0  # Initial temperature at x=0
t2 = 1  # Initial temperature at x=1
end_time = 1  # End time

# + [code]
def pde(x, T):
    dT_xx = dde.grad.hessian(T, x, j=0)
    dT_yy = dde.grad.hessian(T, x, j=1) 
    dT_t = dde.grad.jacobian(T, x, j=2)
    return dT_t - alpha * (dT_xx + dT_yy)

def boundary_x_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundary_x_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)

def boundary_y_b(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0)

def boundary_y_u(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1)

def boundary_initial(x, on_initial):
    return on_initial and np.isclose(x[2], 0)

# + [code]
def init_func(x):
    x_coord = x[:, 0:1]
    t = np.zeros((len(x), 1))
    for i, x_ in enumerate(x_coord):
        if x_ < 0.5:
            t[i] = t1
        else:
            t[i] = t1 + 2 * (x_ - 0.5)
    return t

def dir_func_l(x):
    return t1 * np.ones((len(x), 1))

def dir_func_r(x):
    return t2 * np.ones((len(x), 1))

def func_zero(x):
    return np.zeros((len(x), 1))

# + [code]
# Define geometry and time domain
geom = dde.geometry.Rectangle([0, 0], [1, 1])
timedomain = dde.geometry.TimeDomain(0, end_time)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Define boundary conditions
bc_l = dde.DirichletBC(geomtime, dir_func_l, boundary_x_l)
bc_r = dde.DirichletBC(geomtime, dir_func_r, boundary_x_r)
bc_u = dde.NeumannBC(geomtime, func_zero, boundary_y_u)
bc_b = dde.NeumannBC(geomtime, func_zero, boundary_y_b)
ic = dde.IC(geomtime, init_func, boundary_initial)

# + [code]
# Create TimePDE problem with real-time monitoring
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_l, bc_r, bc_u, bc_b, ic],
    num_domain=30000,
    num_boundary=8000,
    num_initial=20000
)

# Define neural network
layer_size = [3] + [60] * 5 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(lambda x, y: abs(y))

# + [code]
# Create model and compile with monitoring
model = dde.Model(data, net)
model.compile("adam", lr=1e-3, loss_weights=[10, 1, 1, 1, 1, 10])

# Custom callback for real-time monitoring
class TrainingMonitor(dde.callbacks.Callback):
    def on_epoch_end(self):
        if self.model.train_state.epoch % 100 == 0:
            logger.info(f"Epoch {self.model.train_state.epoch}: "
                       f"Loss = {self.model.train_state.loss:.6f}")

# + [code]
# Train model with monitoring
checker = dde.callbacks.ModelCheckpoint(
    "model/model.ckpt", save_better_only=True, period=1000
)
monitor = TrainingMonitor()
logger.info("Starting Adam optimization...")
with tqdm(total=10000, desc="Training (Adam)") as pbar:
    losshistory, train_state = model.train(
        iterations=10000,  # Using iterations instead of deprecated epochs
        batch_size=256,
        callbacks=[checker, monitor],
        display_every=100,
        disregard_previous_best=True
    )
    pbar.update(10000)

# + [code]
# L-BFGS optimization with monitoring
logger.info("Starting L-BFGS optimization...")
model.compile("L-BFGS-B")
dde.optimizers.set_LBFGS_options(maxcor=50)
with tqdm(total=10000, desc="Training (L-BFGS)") as pbar:
    losshistory, train_state = model.train(
        iterations=10000,  # Using iterations instead of deprecated epochs
        batch_size=256,
        display_every=100
    )
    pbar.update(10000)

# + [code]
# Save and plot results
save_path = "results/xde/"
os.makedirs(save_path, exist_ok=True)
dde.saveplot(losshistory, train_state, issave=True, isplot=True, save_path=save_path)

plt.figure(figsize=(10, 6))
plt.semilogy(losshistory.steps, losshistory.loss_train, label="Training loss")
plt.semilogy(losshistory.steps, losshistory.loss_test, label="Testing loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
plt.savefig(os.path.join(save_path, f"deep_xde_baseline_{timestamp}.png"))
plt.show()

logger.info(f"Training completed. Final loss: {train_state.loss:.6f}")
logger.info(f"Results saved to {save_path}")
