import pytest
import tensorflow as tf

import util.tensorplow as tp
from core.augmentor import AudioAugmentor


@pytest.fixture
def spectrogram():
    wav = tp.load_wav('tests/data/tensorplow/test.wav')
    return tp.spectrogram(wav.audio, 48000, 0.01, 0.5)


@pytest.fixture
def frame():
    return tp.load_image('tests/data/tensorplow/test.jpg')


@pytest.fixture
def audio_augmentor():
    return AudioAugmentor((257, 200))


def test_output_should_be_different(spectrogram, audio_augmentor):
    augmented_spectrogram = audio_augmentor(spectrogram)
    assert augmented_spectrogram != spectrogram


def test_output_shape(spectrogram, audio_augmentor):
    augmented_spectrogram = audio_augmentor(spectrogram)
    assert augmented_spectrogram.shape == (257, 200, 1)


def test_with_invalid_input_shape(frame, audio_augmentor):
    with pytest.raises(ValueError):
        assert audio_augmentor(frame)


def test_with_invalid_input_dims(spectrogram, audio_augmentor):
    with pytest.raises(ValueError):
        assert audio_augmentor(tp.run(tf.expand_dims(spectrogram, 0)))
