import pytest
import tensorflow as tf

from core.augmentor import Augmentor, Aug


class Add(Aug):
    def __init__(self, x):
        self.x = x

    def __call__(self, tensor):
        return tf.add(tensor, self.x)


class Scale(Aug):
    def __init__(self, x):
        self.x = x

    def __call__(self, tensor):
        return tf.multiply(tensor, self.x)


@pytest.fixture
def augmentor():
    return Augmentor(Scale(2), Add(1), Add(0))


@pytest.fixture
def empty_augmentor():
    return Augmentor()


@pytest.fixture
def sample_input():
    return tf.random.uniform((10, 10), maxval=10, dtype=tf.dtypes.int32)


def test_augmentor(augmentor, sample_input):
    assert augmentor(sample_input)


def test_non_aug_augmentors():
    with pytest.raises(TypeError):
        assert Augmentor(1)


def test_len(augmentor):
    assert len(augmentor) == 8


def test_output_len(augmentor):
    assert len(augmentor) == 8


def test_empty_augmentor(empty_augmentor, sample_input):
    assert empty_augmentor(sample_input)


def test_len_empty_augmentor(empty_augmentor):
    assert len(empty_augmentor) == 1


def test_output_len_empty_augmentor(empty_augmentor, sample_input):
    assert len(empty_augmentor(sample_input)) == 1
