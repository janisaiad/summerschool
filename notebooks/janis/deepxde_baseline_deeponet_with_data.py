"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import sys

# we add path for your data
sys.path.append('/home/janis/SCIML/summerschool')

kappa = 0.01
c = 0.5

def equation(x, y, f):
    dy_xx = dde.grad.hessian(y, x)
    dy_dx = dde.grad.jacobian(y, x)
    return -kappa*dy_xx + c*dy_dx - f

# Domain is interval [0, 1]
geom = dde.geometry.Interval(0, 1)

def u_boundary(_):
    return 0

def boundary(_, on_boundary):
    return on_boundary

bc = dde.icbc.DirichletBC(geom, u_boundary, boundary)

# AJOUT DE VOTRE DATASET AVANT LA DÉFINITION DU PDE
folder_path = "/home/janis/SCIML/summerschool/data/benchmarks/given/"
x_coords = np.load(folder_path + 'x.npy')
f_train_data = np.load(folder_path + 'f_train_data.npy')
u_train_data = np.load(folder_path + 'u_train_data_c0.5.npy')

print(f"Data shapes: x {x_coords.shape}, f {f_train_data.shape}, u {u_train_data.shape}")

# MODIFICATION CRITIQUE: Adapter le nombre de points de domaine à vos données
# we set num_domain to match your data (100 points) minus boundary points
pde = dde.data.PDE(geom, equation, bc, num_domain=98, num_boundary=2)  # 98 + 2 = 100

# we create custom function space with your data
class YourDataSpace:
    def __init__(self, functions_data, x_coords):
        self.functions = functions_data
        self.x_coords = x_coords
        self.n_functions = len(functions_data)
    
    def random(self, n):
        indices = np.random.choice(self.n_functions, size=n, replace=False)
        return self.functions[indices]
    
    def eval_batch(self, features, points):
        """
        CRITIQUE: nous devons interpoler vos fonctions f aux points demandés par DeepXDE
        """
        # we get the points where DeepXDE wants to evaluate f
        points_flat = points.flatten()
        
        # we interpolate your f functions to these points
        interpolated_features = []
        for f_func in features:
            # we interpolate each function from your grid to DeepXDE's points
            f_interp = np.interp(points_flat, self.x_coords, f_func)
            interpolated_features.append(f_interp)
        
        return np.array(interpolated_features)

space = YourDataSpace(f_train_data[:1000], x_coords)

# CORRECTION: Utiliser les points générés par DeepXDE, pas vos coordonnées x
# DeepXDE va générer ses propres points d'évaluation
num_eval_points = 100  # we keep this to match your data size
evaluation_points = geom.uniform_points(num_eval_points, boundary=True)

# Define PDE operator with your data
pde_op = dde.data.PDEOperatorCartesianProd(
    pde,
    space,
    evaluation_points,
    num_function=200,
)

# Setup DeepONet
dim_x = 1
p = 32
net = dde.nn.DeepONetCartesianProd(
    [num_eval_points, 32, p],
    [dim_x, 32, p],
    activation="tanh",
    kernel_initializer="Glorot normal",
)

# Define and train model
model = dde.Model(pde_op, net)
dde.optimizers.set_LBFGS_options(maxiter=1000)
model.compile("L-BFGS")
model.train()

# Test with your actual data
n_test = 3
test_indices = np.random.choice(len(f_train_data), size=n_test, replace=False)
test_f = f_train_data[test_indices]
test_u_true = u_train_data[test_indices]

# we predict using DeepXDE's evaluation points
x_test = evaluation_points
predictions = []
for i in range(n_test):
    pred = model.predict((test_f[i:i+1], x_test))
    predictions.append(pred.flatten())

# we plot comparison - interpolating results back to your grid
fig, axes = plt.subplots(2, n_test, figsize=(15, 8))

for i in range(n_test):
    # Plot your source functions f(x)
    axes[0, i].plot(x_coords, test_f[i], 'b-', linewidth=2)
    axes[0, i].set_title(f'Your f(x) - Test {i+1}')
    axes[0, i].grid(True)
    
    # Interpolate predictions back to your x coordinates for comparison
    pred_interp = np.interp(x_coords, x_test.flatten(), predictions[i])
    
    # Plot comparison: your true solutions vs predictions
    axes[1, i].plot(x_coords, test_u_true[i], 'r-', label='Your true u(x)', linewidth=2)
    axes[1, i].plot(x_coords, pred_interp, 'g--', label='DeepONet pred', linewidth=2)
    axes[1, i].set_title(f'Solutions u(x) - Test {i+1}')
    axes[1, i].legend()
    axes[1, i].grid(True)

plt.tight_layout()
plt.show()

# we compute errors with your data
for i in range(n_test):
    pred_interp = np.interp(x_coords, x_test.flatten(), predictions[i])
    mse = np.mean((test_u_true[i] - pred_interp)**2)
    print(f"Test {i+1} MSE vs your data: {mse:.6f}")

print("Training with your data completed!")
