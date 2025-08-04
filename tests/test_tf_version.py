import pytest
import tensorflow as tf


def test_tf_version():

    print(tf.__version__)
    assert tf.__version__ >= '2.16.0'


if __name__ == '__main__':
    test_tf_version()