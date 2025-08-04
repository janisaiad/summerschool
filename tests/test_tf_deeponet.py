import pytest

def test_tf_model():
    from model.deeponet.deeponet import DeepONet
    import tensorflow as tf
    
    hyper_params = {"d_p":20,"d_V":20,"learning_rate":0.001,"optimizer":"adam","n_epochs":100,"batch_size":32,"verbose":1,"loss_function":"mse"}
    
    
    regular_params = {
        "internal_model": tf.keras.Sequential([
            tf.keras.layers.Dense(units=20, activation='relu')
        ]),
        "external_model": tf.keras.Sequential([
            tf.keras.layers.Dense(units=20, activation='relu')
        ])
    }
    

    model = DeepONet(hyper_params=hyper_params,regular_params=regular_params)