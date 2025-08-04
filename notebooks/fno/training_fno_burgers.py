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

from sciml.model.fno import FNO
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.config.list_physical_devices('GPU')
import os
from datetime import datetime
# +
nb_xi = 2 # can be deduced from mu because it's len(mu.shape) - 1
p_1 = 30 # dimension of scheme for xi for all i
p_2 = 30 # dimension of scheme for xi for all i
p_3 = 30 # dimension of scheme for xi for all i
epochs = 50
index = 450
n_modes = p_1
n_layers = 4 # need to be low because there is a low difference between the with time
alpha = 0.5
best_loss = 0.00005
activation = 'relu'
kernel_initializer = 'he_normal'
device = "GPU"
n_epochs = epochs



# +
# inputs are of the form [batch, p_1, p_1, nd_xi +1] for nb_xi=2 (+1 because of the mu=f(x))

first_network = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(p_1, p_1,)),  # [batch, p_1, p_1, 3]
    tf.keras.layers.Flatten(),  # [batch, p_1*p_1*3]
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(p_1 * p_1, activation='relu'),
    tf.keras.layers.Reshape((p_1, p_1,))  # [batch, p_1, p_1, p_2]
])

last_network = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(p_1, p_1,)),  # [batch, p_1, p_1, 3]
    tf.keras.layers.Flatten(),  # [batch, p_1*p_1*3]
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(p_1 * p_1 * (1), activation='linear'),
    tf.keras.layers.Reshape((p_1, p_1,))  # [batch, p_1, p_1, 3]
])


# first network graph:
# [batch, p_1, p_1, 3] -> [batch, p_1*p_1*3] -> [batch, 512] -> [batch, 256] -> [batch, p_1*p_1*p_2] -> [batch, p_1, p_1, p_2]
# last network graph:  
# [batch, p_1, p_1, 3] -> [batch, p_1*p_1*3] -> [batch, 256] -> [batch, 512] -> [batch, p_1*p_1*3] -> [batch, p_1, p_1, 3]


# -

folder_path = "data/test_data/example_data/heat2d/"
model = FNO(regular_params={"first_network": first_network, "last_network": last_network},fourier_params={"n_layers": n_layers, "dim_coords":2, "n_modes": n_modes, "activation": activation, "kernel_initializer": kernel_initializer}, hyper_params={"p_1": p_1, "p_2": p_2,'p_3':p_3,"device": device,"n_epochs":n_epochs,"index":index,"alpha":alpha,"folder_path":folder_path,"best_loss":best_loss})

os.makedirs('results/fnograph', exist_ok=True)


date = datetime.now().strftime("%Y%m%d_%H%M%S")

'''
tf.keras.utils.plot_model(model, 
                         to_file=f'results/fno/model_graph_{date}.png',
                         show_shapes=True, 
                         show_layer_names=True)
'''
# we create a txt file to save the model summary

    
# print("Model visualization and summary saved to results/fno/")
# +
# mus, sol = model.get_data_partial(folder_path,alpha=alpha)

# +
# print(mus.shape)
# print(sol.shape)
# -

# import logging
tf.get_logger().setLevel('ERROR')
# Ajouter en haut du notebook pour désactiver tout le logging
# logging.getLogger().setLevel(logging.ERROR)  # Ne montrera que les erreurs graves

loss_history_train,loss_history_test = model.fit_partial(save_weights=True)


try:
    with open(f'results/fnograph/model_summary_{date}.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
except Exception as e:
    print(f"Error saving model summary: {e}")
    pass
    
print(len(loss_history_train))
print(len(loss_history_test))

from datetime import datetime
plt.plot(loss_history_train,color='blue')
plt.plot(loss_history_test,color='red')
plt.legend(['Train','Test'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid()
date = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'results/loss_history_fno{date}.png')
plt.show()


