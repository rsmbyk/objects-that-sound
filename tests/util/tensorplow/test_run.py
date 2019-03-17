import tensorflow as tf

from util import tensorplow as tp


def test_run():
    test_tensor = tf.multiply(2, 3)
    assert tp.run(test_tensor) == 6
