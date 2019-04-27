import numpy as np
import pytest

from core import ops


@pytest.fixture
def spectrogram():
    return 'tests/data/ops/spectrogram.npz'


@pytest.fixture
def non_existing_spectrogram():
    return 'tests/data/ops/missing.npz'


def test_load_spectrogram(spectrogram):
    assert ops.load_spectrogram(spectrogram) is not None


def test_load_non_existing_file(non_existing_spectrogram):
    with pytest.raises(FileNotFoundError):
        assert ops.load_spectrogram(non_existing_spectrogram)


def test_value(spectrogram):
    npz = np.load(spectrogram)['spectrogram']
    spc = ops.load_spectrogram(spectrogram)

    for i in range(spc.shape[0]):
        for j in range(spc.shape[1]):
            assert spc[i, j, 0] == npz[i][j]


def test_shape(spectrogram):
    assert ops.load_spectrogram(spectrogram).shape == (257, 199, 1)
