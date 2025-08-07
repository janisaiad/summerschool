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
pde = dde.data.PDE(geom, equation, bc, num_domain=100, num_boundary=2)

# AJOUT DE VOTRE DATASET
folder_path = "/home/janis/SCIML/summerschool/data/benchmarks/given/"
x_coords = np.load(folder_path + 'x.npy')
f_train_data = np.load(folder_path + 'f_train_data.npy')
u_train_data = np.load(folder_path + 'u_train_data_c0.5.npy')

print(f"Data shapes: x {x_coords.shape}, f {f_train_data.shape}, u {u_train_data.shape}")

# we create custom function space with your data
class YourDataSpace:
    def __init__(self, functions_data):
        self.functions = functions_data
        self.n_functions = len(functions_data)
    
    def random(self, n):
        indices = np.random.choice(self.n_functions, size=n, replace=False)
        return self.functions[indices]
    
    def eval_batch(self, features, points):
        return features

space = YourDataSpace(f_train_data[:1000])  # we use subset
evaluation_points = x_coords.reshape(-1, 1)
num_eval_points = len(evaluation_points)

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

# we predict
x_test = evaluation_points
predictions = []
for i in range(n_test):
    pred = model.predict((test_f[i:i+1], x_test))
    predictions.append(pred.flatten())

# we plot comparison with your real data
fig, axes = plt.subplots(2, n_test, figsize=(15, 8))

for i in range(n_test):
    # Plot your source functions f(x)
    axes[0, i].plot(x_coords, test_f[i], 'b-', linewidth=2)
    axes[0, i].set_title(f'Your f(x) - Test {i+1}')
    axes[0, i].grid(True)
    
    # Plot comparison: your true solutions vs predictions
    axes[1, i].plot(x_coords, test_u_true[i], 'r-', label='Your true u(x)', linewidth=2)
    axes[1, i].plot(x_coords, predictions[i], 'g--', label='DeepONet pred', linewidth=2)
    axes[1, i].set_title(f'Solutions u(x) - Test {i+1}')
    axes[1, i].legend()
    axes[1, i].grid(True)

plt.tight_layout()
plt.show()

# we compute errors with your data
for i in range(n_test):
    mse = np.mean((test_u_true[i] - predictions[i])**2)
    print(f"Test {i+1} MSE vs your data: {mse:.6f}")

print("Training with your data completed!")

# MODIFICATION 4: Ajouter données supervisées pour l'entraînement
class MixedPDEOperator:
    def __init__(self, pde_op, f_data, u_data, x_coords):
        self.pde_op = pde_op
        self.f_data = f_data[:500]  # we use subset for training
        self.u_data = u_data[:500] 
        self.x_coords = x_coords
        
    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        """we combine PDE loss + data loss"""
        # PDE loss from original operator
        pde_losses = self.pde_op.losses(targets, outputs, loss_fn, inputs, model, aux)
        
        # Data loss: we compare predictions with your actual solutions
        f_batch, x_batch = inputs[0], inputs[1]
        u_pred = outputs
        
        # we select corresponding true solutions
        batch_size = len(f_batch)
        indices = np.random.choice(len(self.f_data), size=batch_size, replace=False)
        u_true = self.u_data[indices]
        
        # we compute supervised loss
        data_loss = loss_fn(u_true, u_pred)
        
        # we combine losses
        total_loss = pde_losses[0] + 0.1 * data_loss  # we adjust weight
        
        return [total_loss, pde_losses[0], data_loss]

# we replace the PDE operator
mixed_pde_op = MixedPDEOperator(pde_op, f_train_data, u_train_data, x_coords)

# we use the mixed operator instead
model = dde.Model(mixed_pde_op, net)