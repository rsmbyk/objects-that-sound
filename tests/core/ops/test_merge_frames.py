import contextlib
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from core import ops
from util import ffmpeg


@contextlib.contextmanager
def temp_dir(path):
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path)


@pytest.fixture
def video():
    return 'tests/data/ops/test.mkv'


@pytest.fixture
def wav():
    return 'tests/data/ops/test.wav'


@pytest.fixture
def non_existing_wav():
    return 'tests/data/ops/missing.wav'


@pytest.fixture
def tempdir():
    tdir = 'tests/.temp'
    os.makedirs(tdir, exist_ok=True)
    return tdir


@pytest.fixture
def frames_dir(tempdir):
    fdir = tempfile.TemporaryDirectory(dir=tempdir).name
    os.makedirs(fdir, exist_ok=True)
    return fdir


@pytest.fixture
def non_existing_frames_dir():
    return 'tests/.temp/ops/missing'


@pytest.fixture
def output():
    return 'tests/.temp/ops/output.mkv'


def extract_to_pngs(raw, output_dir):
    ffmpeg.ffmpeg(raw, os.path.join(output_dir, '%d.png'),
                  r=25,
                  start_number=0,
                  vf='scale=256:256:force_original_aspect_ratio=increase')


def test_merge_frames(video, wav, frames_dir, output):
    with temp_dir(frames_dir):
        extract_to_pngs(video, frames_dir)
        ops.merge_frames(frames_dir, wav, output)
        assert os.path.exists(output)
        os.remove(output)


def test_output_video_duration(video, wav, frames_dir, output):
    with temp_dir(frames_dir):
        extract_to_pngs(video, frames_dir)
        ops.merge_frames(frames_dir, wav, output)
        assert abs(ops.get_video_duration(output) - ops.get_video_duration(video)) < 1
        os.remove(output)


def test_merge_frames_with_non_existing_frames_dir(wav, non_existing_frames_dir, output):
    with pytest.raises(FileNotFoundError):
        ops.merge_frames(non_existing_frames_dir, wav, output)


def test_merge_frames_with_non_dir_frames_dir(video, wav, output):
    with pytest.raises(NotADirectoryError):
        ops.merge_frames(video, wav, output)


def test_merge_frames_with_non_existing_wav(video, non_existing_wav, frames_dir, output):
    with temp_dir(frames_dir):
        with pytest.raises(FileNotFoundError):
            extract_to_pngs(video, frames_dir)
            ops.merge_frames(frames_dir, non_existing_wav, output)


def test_merge_frames_with_existing_output(video, wav, frames_dir, output):
    with temp_dir(frames_dir):
        extract_to_pngs(video, frames_dir)
        Path(output).touch()
        assert os.path.exists(output)
        ops.merge_frames(frames_dir, wav, output)
        assert os.path.exists(output)
        os.remove(output)
