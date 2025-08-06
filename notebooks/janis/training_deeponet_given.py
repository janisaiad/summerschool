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

# %%
d_p = 10
d_V = 10
epochs = 100 


# %%
internal_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(100,)),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(40, activation='sigmoid'),
    tf.keras.layers.Dense(40, activation='relu'),
])


external_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(1,)),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(40, activation='sigmoid'),
    tf.keras.layers.Dense(40, activation='relu'),
])

# %%
folder_path = "/home/janis/SCIML/summerschool/data/benchmarks/given/"

# %%
model = DeepONet(regular_params={"internal_model": internal_model, "external_model": external_model}, hyper_params={"d_p": d_p, "d_V": d_V,"device": "GPU","n_epochs":epochs},folder_path="../../data/benchmarks/given/")

# %%
mus, xs, sol = get_mu_xs_sol(folder_path,0.2)

# %%
print(mus.shape)
print(xs.shape)
print(sol.shape)

# %%
train_history = model.fit()

# %%
print(train_history)

# %%
plt.figure(figsize=(10,6))
plt.plot(train_history[0], label='Training Loss')  # we plot training loss
plt.plot(train_history[1], label='Test Loss')  # we plot test loss
plt.yscale('log')  # we use log scale for better visualization
plt.grid(True, which="both", ls="-", alpha=0.2)  # we add grid with transparency
plt.xlabel('Epochs')  # we add x label
plt.ylabel('Loss')  # we add y label
plt.title('Training and Test Loss Over Time')  # we add title
plt.legend()  # we add legend
plt.yscale('log')
plt.xscale('log')
plt.show()

# %%
# then we can test the model
mu_test, xs_test, sol_test = get_mu_xs_sol(folder_path,0.2,training=False)

preds = model.predict(mu_test, xs_test)
for i in range(10):
    plt.plot(xs_test[i,:],sol_test[i,:], label='True')
    plt.plot(xs_test[i,:],preds[i,:], label='Predicted')
    plt.legend()
    plt.show()

# %%
