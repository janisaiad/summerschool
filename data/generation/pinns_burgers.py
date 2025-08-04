import numpy as np
import tensorflow as tf
import os

class BurgersPINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(50, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(50, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(50, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(2)
        
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)

@tf.function
def pde_loss(model, x):
    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            u = model(x)
        du_dx = tape1.gradient(u, x)
        du_dt = du_dx[:,:1]
        du_dx = du_dx[:,1:2]
        du_dy = du_dx[:,2:]
        
    f_u = du_dt + du_dx + du_dy
    return tf.reduce_mean(tf.square(f_u))

data = np.load("/home/janis/SCIML/sciml/data/test_data/example_data/burgers2d/sol/sol_0.npy")
points = np.load("/home/janis/SCIML/sciml/data/test_data/example_data/burgers2d/xs/xs_0.npy")

x_train = tf.convert_to_tensor(points, dtype=tf.float32)
y_train = tf.convert_to_tensor(data.reshape(-1, 2), dtype=tf.float32)

model = BurgersPINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        y_pred = model(x_batch)
        mse_loss = tf.reduce_mean(tf.square(y_pred - y_batch))
        physics_loss = pde_loss(model, x_batch)
        total_loss = mse_loss + physics_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss

batch_size = 1024
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)

for epoch in range(10000):
    for x_batch, y_batch in dataset:
        loss = train_step(x_batch, y_batch)

