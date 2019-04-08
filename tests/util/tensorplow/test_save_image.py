import os

import pytest
import tensorflow as tf

from util import tensorplow as tp


@pytest.fixture
def output():
    return 'tests/.temp/tensorplow/out.png'


def test_save_image(output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    image = tf.random.uniform((100, 100, 3))
    tp.save_image(tp.run(image), output)
    assert os.path.exists(output)
    os.remove(output)
    os.removedirs(os.path.dirname(output))


def test_save_single_channel_image(output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    image = tf.random.uniform((100, 100, 1))
    tp.save_image(tp.run(image), output)
    assert os.path.exists(output)
    os.remove(output)
    os.removedirs(os.path.dirname(output))


def test_should_create_output_parent_dir(output):
    image = tf.random.uniform((100, 100, 1))
    tp.save_image(tp.run(image), output)
    assert os.path.exists(os.path.dirname(output))
    os.remove(output)
    os.removedirs(os.path.dirname(output))
