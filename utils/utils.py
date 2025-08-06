import numpy as np
import tensorflow as tf
import os
import json

def get_data(folder_path: str, alpha: float = 0.01):
    try:
        with open(os.path.join(folder_path, "params.json"), "r") as f:
            params = json.load(f)
    
        nx = params["nx"]
        ny = params["ny"]
        nt = params["nt"] 
        n_mu = params["n_mu"]
        
        mu_list = []
        sol_list = []
        
        for i in range(int(n_mu * alpha)):
            mu_list.append(np.load(os.path.join(folder_path, f"mu/mu_{i}.npy")))
            sol_list.append(np.load(os.path.join(folder_path, f"sol/sol_{i}.npy")))
        
        mu = np.stack(mu_list, axis=0)
        sol = np.stack(sol_list, axis=0)
            
        mu = tf.convert_to_tensor(mu, dtype=tf.float32)
        sol = tf.convert_to_tensor(sol, dtype=tf.float32)
        
        inputs = tf.squeeze(mu, axis=-1)
        
        return inputs, sol
        
    except Exception as e:
        raise ValueError(f"Failed to load data: {str(e)}")

def compute_rademacher_complexity(inputs: tf.Tensor, n_samples: int = 1000):
    sigma = tf.random.uniform(shape=[n_samples], minval=0, maxval=2, dtype=tf.int32) * 2 - 1
    sigma = tf.cast(sigma, tf.float32)
    sigma_expanded = tf.reshape(sigma, [n_samples, 1, 1, 1])
    complexity = tf.reduce_mean(tf.abs(
        tf.reduce_mean(tf.multiply(sigma_expanded, inputs), axis=0)
    ))
    return float(complexity)

def compute_kolmogorov_width(inputs: tf.Tensor, n: int = 10):
    flattened = tf.reshape(inputs, [inputs.shape[0], -1])
    s = tf.linalg.svd(flattened, compute_uv=False)
    width = tf.reduce_sum(s[n:])
    return float(width)

if __name__ == "__main__":
    inputs, _ = get_data("data/test_data/big_dataset_fno/heat2d/", alpha=0.01)
    
    os.makedirs("results/rademacher_fno_big", exist_ok=True)
    with open("results/rademacher_fno_big/complexity.txt", "w") as f:
        f.write(str(compute_rademacher_complexity(inputs)) + "\n")
        f.write(str(compute_kolmogorov_width(inputs)))






@tf.function
def custom_pinn_loss_2nd_order(model:tf.keras.Model,mu:tf.Tensor,x:tf.Tensor,sol:tf.Tensor,coeff:tf.Tensor)->tf.Tensor:
    
    """ I just make a custome PINN type loss function, to use it then with deeponet"""
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x) # need to watch bc wrt to this input
        
        y_pred = model.predict(mu,x)    
        dydx = tape.gradient(y_pred,x)
        dydx = tf.squeeze(dydx,axis=-1) # because we have a 1D problem since x is a tensor
        dydx = tf.cast(dydx,dtype=y_pred.dtype)
        d2ydx2 = tape.gradient(dydx,x)
        d2ydx2 = tf.squeeze(d2ydx2,axis=-1) # because we have a 1D problem
        d2ydx2 = tf.cast(d2ydx2,dtype=y_pred.dtype) # because we have a 1D problem
    del tape
    terms  = tf.stack([d2ydx2,dydx,y_pred],axis=-1)
    pde_terms = tf.reduce_sum(terms*coeff,axis=-1)
    pinn_loss = tf.reduce_mean(tf.square(pde_terms-tf.cast(mu,dtype=y_pred.dtype))) # because we have a PDE of the form d2ydx2 + dydx + y = mu    
    # now we enforce the boundary conditions
    boundary_loss = tf.reduce_mean(tf.square(y_pred[:,0]))
    boundary_loss += tf.reduce_mean(tf.square(y_pred[:,-1]))
    pinn_loss += 0.00*boundary_loss
    return pinn_loss
