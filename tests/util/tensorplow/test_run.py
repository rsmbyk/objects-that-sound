import tensorflow as tf

from util import tensorplow as tp


def test_run():
    test_tensor = tf.multiply(2, 3)
    assert tp.run(test_tensor) == 6


def test_run_with_feed_dict():
    x = tf.placeholder(tf.dtypes.uint8)
    y = tf.placeholder(tf.dtypes.uint8)
    multiply = tf.multiply(x, y)
    assert tp.run(multiply, {x: 2, y: 3}) == 6
