import pytest
import tensorflow as tf

import util.tensorplow as tp
from core.augmentor import VisionAugmentor


@pytest.fixture
def frame():
    return tp.load_image('tests/data/tensorplow/test.jpg')


@pytest.fixture
def spectrogram():
    wav = tp.load_wav('tests/data/tensorplow/test.wav')
    return tp.spectrogram(wav.audio, 48000, 0.01, 0.5)


@pytest.fixture
def vision_augmentor():
    return VisionAugmentor((224, 224, 3))


def test_output_should_be_different(frame, vision_augmentor):
    augmented_image = vision_augmentor(frame)
    assert augmented_image != frame


def test_output_shape(frame, vision_augmentor):
    augmented_image = vision_augmentor(frame)
    assert augmented_image.shape == (224, 224, 3)


def test_with_invalid_input_shape(spectrogram, vision_augmentor):
    with pytest.raises(ValueError):
        assert vision_augmentor(spectrogram)


def test_with_invalid_input_dims(frame, vision_augmentor):
    with pytest.raises(ValueError):
        assert vision_augmentor(tp.run(tf.expand_dims(frame, 0)))
