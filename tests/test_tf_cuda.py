import pytest


def test_tf_cuda():
    import tensorflow as tf
    print(tf.config.list_physical_devices())
    gpus = tf.config.list_physical_devices('GPU')
    assert len(gpus) > 0


if __name__ == "__main__":
    test_tf_cuda()