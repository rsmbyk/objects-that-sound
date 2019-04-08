import pytest

from util import tensorplow as tp


@pytest.fixture
def test_wav_file():
    return 'tests/data/tensorplow/test.wav'


@pytest.fixture
def non_existing_test_wav_file():
    return 'tests/data/tensorplow/missing.jpg'


def test_load_wav(test_wav_file):
    wav = tp.load_wav(test_wav_file)
    assert wav.sample_rate == 48000
    assert len(wav.audio) >= 48000


def test_load_non_existing_wav(non_existing_test_wav_file):
    with pytest.raises(FileNotFoundError):
        tp.load_wav(non_existing_test_wav_file)
