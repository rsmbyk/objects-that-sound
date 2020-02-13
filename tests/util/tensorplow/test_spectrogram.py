import pytest

from util import tensorplow as tp


@pytest.fixture
def test_wav_file():
    return 'tests/data/tensorplow/test.wav'


def test_spectrogram(test_wav_file):
    wav = tp.load_wav(test_wav_file)
    spc = tp.spectrogram(wav.audio, 48000, 0.01, 0.5)
    assert spc.shape == (257, 199)
