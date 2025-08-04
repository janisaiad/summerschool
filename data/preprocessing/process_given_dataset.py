import numpy as np
import tensorflow as tf

import os

def get_mu_xs_sol(folder_path,type, training=True): # we get that from the given dataset that is like x shape: (100,) f_train shape: (2000, 100) f_test shape: (1000, 100) u_test_type1 shape: (1000, 100) u_test_type2 shape: (1000, 100) u_test_type3 shape: (1000, 100)
    """Type is the type of the solution we want to get, it can be 0.2, 0.5, 1.0"""
    
    x = np.load(folder_path + 'x.npy')
    if training:
        f_train = np.load(folder_path + 'f_train_data.npy')
        u_train_type = np.load(folder_path + f'u_train_data_c{type}.npy')
    else:
        f_test = np.load(folder_path + 'f_test_data.npy')
        u_test_type = np.load(folder_path + f'u_test_data_c{type}.npy')
    
    if training:
        mus = f_train
    else:
        mus = f_test
    
    if training:
        xs = tf.tile(tf.expand_dims(x, 0), [f_train.shape[0], 1]) # we create a tensor of size (n_sol, n_points) by repeating x for each solution
    else:
        xs = tf.tile(tf.expand_dims(x, 0), [f_test.shape[0], 1]) # we create a tensor of size (n_sol, n_points) by repeating x for each solution
    xs = tf.expand_dims(xs, -1) # we add a dimension for the spatial dimension to get (n_sol, n_points, 1)
    
    if training:
        sol = u_train_type
    else:
        sol = u_test_type
    
    mus = tf.convert_to_tensor(mus)
    xs = tf.convert_to_tensor(xs)
    sol = tf.convert_to_tensor(sol)
    
    print("mus.shape", mus.shape)
    print("xs.shape", xs.shape)
    print("sol.shape", sol.shape)
    
    return mus, xs, sol


if __name__ == "__main__":
    folder_path = "data/benchmarks/given/"
        
    for type in [0.2, 0.5, 1.0]:
        mus, xs, sol = get_mu_xs_sol(folder_path,type)
        
    mus, xs, sol = get_mu_xs_sol(folder_path,0.2)
    print("mus.shape", mus.shape)
    print("xs.shape", xs.shape)
    print("sol.shape", sol.shape)
    