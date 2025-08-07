"""
DeepONet avec PINN loss pour l'équation de diffusion-advection 1D
Équation: -κu''(x) + cu'(x) = f(x) avec κ = 0.01 et c > 0
Conditions aux limites: u(0) = u(1) = 0
"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sys

# we add the parent directory to import our custom functions
sys.path.append('/home/janis/SCIML/summerschool')
from data.preprocessing.process_given_dataset import get_mu_xs_sol

# we define the parameters for the advection-diffusion equation
kappa = 0.01  # diffusion coefficient
c = 0.5       # advection coefficient (you can modify this)

def equation(x, y, f):
    """
    we define the advection-diffusion equation: -κu''(x) + cu'(x) = f(x)
    """
    dy_dx = dde.grad.jacobian(y, x)    # first derivative u'(x)
    dy_xx = dde.grad.hessian(y, x)     # second derivative u''(x)
    
    # we return the PDE residual: -κu''(x) + cu'(x) - f(x) = 0
    return -kappa * dy_xx + c * dy_dx - f

# we define the domain as interval [0, 1]
geom = dde.geometry.Interval(0, 1)

# we define zero Dirichlet boundary conditions
def u_boundary(_):
    return 0

def boundary(_, on_boundary):
    return on_boundary

bc = dde.icbc.DirichletBC(geom, u_boundary, boundary)

# we define the PDE with more domain points for better accuracy
pde = dde.data.PDE(geom, equation, bc, num_domain=200, num_boundary=2)

# we load your data from the given dataset
folder_path = "/home/janis/SCIML/summerschool/data/benchmarks/given/"
type_value = 0.5  # we choose c = 0.5 as example

# we get training data
mus_train, xs_train, sol_train = get_mu_xs_sol(folder_path, type_value, training=True)
mus_test, xs_test, sol_test = get_mu_xs_sol(folder_path, type_value, training=False)

# we convert to numpy arrays immediately to avoid symbolic tensor issues
mus_train_np = mus_train.numpy()
xs_train_np = xs_train.numpy()
sol_train_np = sol_train.numpy()
mus_test_np = mus_test.numpy()
xs_test_np = xs_test.numpy()
sol_test_np = sol_test.numpy()

print(f"Training data shapes: mus {mus_train_np.shape}, xs {xs_train_np.shape}, sol {sol_train_np.shape}")
print(f"Test data shapes: mus {mus_test_np.shape}, xs {xs_test_np.shape}, sol {sol_test_np.shape}")

# we create evaluation points from your data
evaluation_points = xs_train_np[0, :, 0].reshape(-1, 1)  # we extract x coordinates
num_eval_points = len(evaluation_points)

# we create a custom function space based on your data
class CustomFunctionSpace:
    def __init__(self, mus_data):
        self.mus_data = mus_data  # already numpy
        self.n_functions = len(self.mus_data)
    
    def random(self, n):
        """we sample random functions from our dataset"""
        indices = np.random.choice(self.n_functions, size=n, replace=False)
        return self.mus_data[indices]
    
    def eval_batch(self, features, points):
        """we evaluate the functions at given points"""
        return features  # in our case, functions are already evaluated at grid points

space = CustomFunctionSpace(mus_train_np)

# we create a custom PDE operator that uses your data
class CustomPDEOperator:
    def __init__(self, pde, space, evaluation_points, num_function):
        self.pde = pde
        self.space = space
        self.evaluation_points = evaluation_points
        self.num_function = num_function
        
        # we generate training data from your dataset
        self.generate_training_data()
    
    def generate_training_data(self):
        """we generate training data from your custom dataset"""
        n_train = min(self.num_function, len(mus_train_np))
        
        # we select random indices for training
        indices = np.random.choice(len(mus_train_np), size=n_train, replace=False)
        
        self.functions = mus_train_np[indices]  # source functions f(x)
        self.locations = xs_train_np[indices]   # spatial coordinates x
        self.solutions = sol_train_np[indices]  # solutions u(x)
        
    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        """we compute both data loss and physics loss"""
        # we get the predicted solutions
        f_pred, x_pred = inputs[0], inputs[1]
        u_pred = outputs
        
        # Data loss (supervised)
        data_loss = loss_fn(targets, u_pred)
        
        # Physics loss (PINN component)
        physics_loss = self.compute_physics_loss(model, f_pred, x_pred)
        
        # we combine losses with weighting
        total_loss = data_loss + 0.01 * physics_loss  # we adjust weight as needed
        
        return [total_loss, data_loss, physics_loss]
    
    def compute_physics_loss(self, model, f_batch, x_batch):
        """we compute the physics-informed loss"""
        physics_losses = []
        
        for i in range(len(f_batch)):
            f_i = f_batch[i:i+1]  # single function
            x_i = x_batch[i]      # corresponding spatial points
            
            # we reshape x for gradient computation
            x_tensor = tf.constant(x_i, dtype=tf.float32)
            x_tensor = tf.reshape(x_tensor, [-1, 1])
            
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_tensor)
                
                # we predict solution at these points
                u_pred = model((f_i, x_tensor))
                
                # we compute gradients
                du_dx = tape.gradient(u_pred, x_tensor)
                
            with tf.GradientTape() as tape2:
                tape2.watch(x_tensor)
                du_dx_watched = tape.gradient(u_pred, x_tensor)
                
            d2u_dx2 = tape2.gradient(du_dx_watched, x_tensor)
            
            del tape
            
            # we compute PDE residual: -κu''(x) + cu'(x) - f(x)
            f_values = tf.reshape(f_i, [-1, 1])
            pde_residual = -kappa * d2u_dx2 + c * du_dx - f_values
            
            physics_loss_i = tf.reduce_mean(tf.square(pde_residual))
            physics_losses.append(physics_loss_i)
        
        return tf.reduce_mean(physics_losses)

# we create the custom PDE operator
pde_op = CustomPDEOperator(pde, space, evaluation_points, num_function=1000)

# we setup DeepONet architecture
dim_x = 1
p = 64  # we increase latent dimension for better representation

net = dde.nn.DeepONetCartesianProd(
    [num_eval_points, 64, 64, p],  # branch network (for functions f(x))
    [dim_x, 64, 64, p],            # trunk network (for coordinates x)
    activation="tanh",
    kernel_initializer="Glorot normal",
)

# we define the model with custom loss handling
class PINNDeepONet(dde.Model):
    def __init__(self, data, net):
        super().__init__(data, net)
        
    def compile(self, optimizer, lr=None, loss="MSE", metrics=None, 
                decay=None, loss_weights=None, external_trainable_variables=None):
        """we override compile to handle multiple losses"""
        super().compile(optimizer, lr, loss, metrics, decay, loss_weights, external_trainable_variables)
        
        # we modify the loss function to handle our custom losses
        if isinstance(self.data, CustomPDEOperator):
            self.data.losses = lambda targets, outputs, loss_fn, inputs, model, aux=None: \
                self.data.losses(targets, outputs, loss_fn, inputs, self.net, aux)

# we create and compile the model
model = PINNDeepONet(pde_op, net)
model.compile("adam", lr=1e-3, loss="MSE")

# we train the model
history = model.train(iterations=5000, display_every=500)

# we test the model on some examples
n_test = 3
test_indices = np.random.choice(len(mus_test_np), size=n_test, replace=False)

# we create test data
test_functions = mus_test_np[test_indices]
test_x = xs_test_np[test_indices]
test_solutions = sol_test_np[test_indices]

# we predict solutions
x_test_points = test_x[0, :, 0].reshape(-1, 1)  # we use same x points for all
predictions = []

for i in range(n_test):
    f_test = test_functions[i:i+1]
    pred = model.predict((f_test, x_test_points))
    predictions.append(pred.flatten())

# we plot results
fig, axes = plt.subplots(2, n_test, figsize=(15, 8))

for i in range(n_test):
    # we plot source functions f(x)
    axes[0, i].plot(x_test_points.flatten(), test_functions[i], 'b-', label='f(x)', linewidth=2)
    axes[0, i].set_title(f'Source function f(x) - Test {i+1}')
    axes[0, i].set_ylabel('f(x)')
    axes[0, i].grid(True)
    axes[0, i].legend()
    
    # we plot solutions u(x)
    axes[1, i].plot(x_test_points.flatten(), test_solutions[i], 'r-', label='True u(x)', linewidth=2)
    axes[1, i].plot(x_test_points.flatten(), predictions[i], 'g--', label='Predicted u(x)', linewidth=2)
    axes[1, i].set_title(f'Solution u(x) - Test {i+1}')
    axes[1, i].set_ylabel('u(x)')
    axes[1, i].set_xlabel('x')
    axes[1, i].grid(True)
    axes[1, i].legend()

plt.tight_layout()
plt.savefig('/home/janis/SCIML/summerschool/results/deepxde_pinn_deeponet_results.png', dpi=300, bbox_inches='tight')
plt.show()

# we compute and print errors
errors = []
for i in range(n_test):
    error = np.mean((test_solutions[i] - predictions[i])**2)
    errors.append(error)
    print(f"Test {i+1} MSE: {error:.6f}")

print(f"Average MSE: {np.mean(errors):.6f}")

# we save the model
model.save('/home/janis/SCIML/summerschool/models/deepxde_pinn_deeponet')
print("Model saved successfully!")
