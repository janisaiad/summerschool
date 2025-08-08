# %%
# the key idea is that instead of doing L2 training we do sobolev training ! 

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from sciml.model.deeponet import DeepONet
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sciml.data.preprocessing.process_given_dataset import get_mu_xs_sol

tf.config.list_physical_devices('GPU')

for d_p,d_V,learning_rate,width in [
    (10,10,0.001,32),
    (10,10,0.001,64), 
    (10,10,0.001,128),
    (10,10,0.0005,32),
    (10,10,0.0005,64), 
    (10,10,0.0005,128),
    (10,10,0.0005,256),
    (20,20,0.001,32),
    (20,20,0.001,64),
    (20,20,0.001,128),
    (20,20,0.001,256),
    (20,20,0.0005,32),
    (20,20,0.0005,64),
    (20,20,0.0005,128),
    (20,20,0.0005,256)]:
    # %%


# %%

# %% [markdown]
# trying to enforce the boundary condition

    # %%
    class ConstrainedExternalModel(tf.keras.Model):
        def __init__(self, base_model, d_V):
            super().__init__()
            self.base_model = base_model
            self.d_V = d_V
        
        def call(self, x):
            """
            External model with boundary constraints: forces solution to be 0 at x=0 and x=1
            Uses basis functions that vanish at boundaries: x*(1-x), x^2*(1-x), etc.
            """
            # we get the base model output for learnable coefficients
            base_output = self.base_model(x)  # shape: (batch*n_points, d_V-1)
            
            # we extract x coordinates 
            x_coords = tf.squeeze(x, axis=-1)  # shape: (batch*n_points,)
            
            # we create basis functions that vanish at boundaries
            # First basis: x*(1-x) - vanishes at both x=0 and x=1
            basis_boundary = tf.expand_dims(x_coords * (1.0 - x_coords), axis=1)
            
            # we multiply all learned bases by x*(1-x) to ensure they vanish at boundaries
            x_factor = tf.expand_dims(x_coords * (1.0 - x_coords), axis=1)
            constrained_bases = base_output * x_factor
            
            # we concatenate: [x*(1-x), constrained_bases...]
            constrained_output = tf.concat([basis_boundary, constrained_bases], axis=1)
            
            return constrained_output
    epochs = 20
    # we create the normal branch model (no constraints needed)
    internal_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(100,)),
        tf.keras.layers.Dense(width, activation='elu'),
        tf.keras.layers.Dense(width, activation='elu'),
        tf.keras.layers.Dense(d_V, activation='elu')
        # full d_V output
    ])

    # we create the constrained trunk model
    base_external = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(1,)),
        tf.keras.layers.Dense(4*width, activation='elu'), # this 4 is arbitrary, it worked fine at a moment
        tf.keras.layers.Dense(2*width, activation='elu'),
        tf.keras.layers.Dense(d_V-1, activation='elu')  # we reduce by 1 for the boundary basis
    ])

    external_model = ConstrainedExternalModel(base_external, d_V)

    # %%

    coeff = tf.constant([-0.01,1,0],dtype=tf.float32)
    model = DeepONet(regular_params={"internal_model": internal_model, "external_model": external_model}, hyper_params={"pinn_order":2,"pinn_coeff": coeff,"d_p": d_p, "d_V": d_V,"device": "GPU","n_epochs":epochs,"learning_rate":0.002},folder_path="/home/janis/SCIML/summerschool/data/benchmarks/given/")

    # %%


    # now we need to do sobolev training with custom function knowing the PDE


    # %%
    folder_path = "/home/janis/SCIML/summerschool/data/benchmarks/given/"
    coeff = tf.constant([-0.01,1,0],dtype=tf.float32)
    # %%
    model = DeepONet(regular_params={"internal_model": internal_model, "external_model": external_model}, hyper_params={"pinn_order":2,"pinn_coeff": coeff,"d_p": d_p, "d_V": d_V,"device": "GPU","n_epochs":epochs},folder_path="/home/janis/SCIML/summerschool/data/benchmarks/given/")

    # %%
    mus, xs, sol = get_mu_xs_sol(folder_path,1)

    # %%
    print(mus.shape)
    print(xs.shape)
    print(sol.shape)

    # %%

    # %%
    # we initialize empty lists to store training and test losses
    train_losses = []
    test_losses = []

    for k in range(10):
        # we get training history and append losses to lists
        train_history = model.fit()
        train_losses+=train_history[0]
        test_losses+=train_history[1]
        
        # we plot training progress
        plt.figure(figsize=(10,6))
        plt.plot(train_losses, label='Training Loss')  # we plot training loss
        plt.plot(test_losses, label='Test Loss')  # we plot test loss
        plt.yscale('log')  # we use log scale for better visualization
        plt.grid(True, which="both", ls="-", alpha=0.2)  # we add grid with transparency
        plt.xlabel('Epochs')  # we add x label
        plt.ylabel('Loss')  # we add y label
        plt.title('Training and Test Loss Over Time')  # we add title
        plt.legend()  # we add legend
        plt.xscale('log')
        
        plt.savefig(f"/home/janis/SCIML/summerschool/notebooks/janis/png/deeponet_pinn_d_p_{d_p}_d_V_{d_V}_width_{width}_learning_rate_{learning_rate}_train_test_loss_epoch_{k}.png")
        plt.close()    
        # we test the model
        mu_test, xs_test, sol_test = get_mu_xs_sol(folder_path,1,training=False)
        preds = model.predict(mu_test, xs_test)
        for i in range(20):
            plt.plot(xs_test[i,:],sol_test[i,:], label='True')
            plt.plot(xs_test[i,:],preds[i,:], label='Predicted')
            plt.legend()
            plt.savefig(f"/home/janis/SCIML/summerschool/notebooks/janis/png/deeponet_pinn_d_p_{d_p}_d_V_{d_V}_width_{width}_learning_rate_{learning_rate}_func_{i}_epoch_{k}.png")
            plt.close()
    # %%
    # we plot the basis functions learned by the DeepONet
    plt.figure(figsize=(12,8))
    branch_weights = model.internal_model.layers[-1].weights[0].numpy()  # we get the weights from the last layer of internal model
    num_basis = branch_weights.shape[-1]  # we get number of basis functions

    # we create input points for visualization 
    x_plot = np.linspace(0, 1, 100)
    x_plot = x_plot.reshape(-1, 1)

    # we get external model outputs for these points
    external_outputs = model.external_model(x_plot)  # we evaluate external model

    # we plot each basis function
    for i in range(num_basis):
        basis_func = external_outputs.numpy()[:, i]  # we extract i-th basis function
        plt.plot(x_plot, basis_func, label=f'Basis {i+1}')  # we plot the basis function

    plt.title('DeepONet Learned Basis Functions')  # we add title
    plt.xlabel('x')  # we add x label
    plt.ylabel('Basis Function Value')  # we add y label
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # we place legend outside
    plt.grid(True, alpha=0.3)  # we add light grid
    plt.tight_layout()  # we adjust layout
    plt.savefig(f"/home/janis/SCIML/summerschool/notebooks/janis/png/deeponet_pinn_d_p_{d_p}_d_V_{d_V}_width_{width}_learning_rate_{learning_rate}_basis_functions.png")



    # %%
