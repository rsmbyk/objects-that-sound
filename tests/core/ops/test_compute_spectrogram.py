import os

import pytest

from core import ops
from util import tensorplow as tp


@pytest.fixture
def test_wav_file():
    return 'tests/data/ops/test.wav'


@pytest.fixture
def output():
    return 'tests/.temp/ops/spectrogram.npz'


def test_compute_spectrogram(test_wav_file, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    wav = tp.load_wav(test_wav_file)
    ops.compute_spectrogram(wav.audio, output)
    assert os.path.exists(output)
    os.remove(output)
    os.removedirs(os.path.dirname(output))


def test_should_skip_already_computed_spectrogram(test_wav_file, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    wav = tp.load_wav(test_wav_file)
    ops.compute_spectrogram(wav.audio, output)
    created_time = os.path.getmtime(output)
    ops.compute_spectrogram(wav.audio, output)
    assert os.path.getmtime(output) == created_time
    os.remove(output)
    os.removedirs(os.path.dirname(output))


def test_should_create_output_parent_dir(test_wav_file, output):
    wav = tp.load_wav(test_wav_file)
    ops.compute_spectrogram(wav.audio, output)
    assert os.path.exists(os.path.dirname(output))
    os.remove(output)
    os.removedirs(os.path.dirname(output))
