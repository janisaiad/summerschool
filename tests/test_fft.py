import tensorflow as tf
import pytest as pt





def test_fft():
    n_modes = 10
    n_points = 5
    batch_size = 3
    
    x = tf.random.uniform((batch_size, n_points,17))
    inputs = tf.random.uniform((batch_size, n_points, 1))
    
    data = tf.concat([x, inputs], axis=-1)
    
    fft = tf.signal.rfftnd(data, axes=[-1], fft_length=[n_modes])
    print(fft.shape)



if __name__ == "__main__":
    test_fft()