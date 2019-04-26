import pytest

from core.augmentor_impl import *


@pytest.fixture
def im():
    return tf.random.uniform((10, 10, 3), maxval=10, dtype=tf.dtypes.float32)


def test_scale(im):
    assert Scale(2)(im) is not None


def test_random_scale(im):
    assert RandomScale(0.1, True)(im) is not None


def test_resize(im):
    assert Resize(15, 15)(im) is not None


def test_random_crop(im):
    assert RandomCrop(5, 5, 3)(im) is not None


def test_random_saturation(im):
    assert RandomSaturation(0.25)(im) is not None


def test_brightness(im):
    assert RandomBrightness(0.5)(im) is not None


def test_random_horizontal_flip(im):
    assert RandomHorizontalFlip()(im) is not None
