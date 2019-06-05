import contextlib
import os
import shutil

import pytest

from util import ffmpeg


@contextlib.contextmanager
def temp_copy(path, to):
    os.makedirs(os.path.dirname(to), exist_ok=True)
    shutil.copy(path, to)
    yield to
    if os.path.isdir(to):
        shutil.rmtree(to)
    else:
        os.remove(to)


@pytest.fixture
def test_infile():
    return 'tests/data/ffmpeg/test.mp4'


@pytest.fixture
def non_existing_test_infile():
    return 'tests/data/ffmpeg/infile'


@pytest.fixture
def test_infile_with_space():
    return 'tests/.temp/ffmpeg/input file/test.mp4'


def test_ffprobe_not_available():
    ffprobe_path = shutil.which('ffprobe')

    if ffprobe_path:
        # remove ffprobe from PATH
        path = os.environ['PATH'].split(';')
        path.remove(os.path.dirname(ffprobe_path))
        os.environ['PATH'] = ';'.join(path)

    assert not shutil.which('ffprobe')

    with pytest.raises(AssertionError):
        ffmpeg.ffprobe(None, None)

    if ffprobe_path:
        # put ffprobe back to PATH
        path = os.environ['PATH'].split(';')
        path.append(os.path.dirname(ffprobe_path))
        os.environ['PATH'] = ';'.join(path)

    assert shutil.which('ffprobe')


def test_ffprobe_with_non_existing_infile(non_existing_test_infile):
    with pytest.raises(FileNotFoundError):
        ffmpeg.ffprobe(non_existing_test_infile)


def test_ffprobe_process_video(test_infile):
    ffmpeg.ffprobe(test_infile)


def test_ffprobe_with_invalid_parameters(test_infile):
    with pytest.raises(RuntimeError):
        ffmpeg.ffprobe(test_infile, 'invalid_flag')
    with pytest.raises(RuntimeError):
        ffmpeg.ffprobe(test_infile, invalid_option=True)


def test_ffprobe_with_path_containing_space(test_infile, test_infile_with_space):
    with temp_copy(test_infile, test_infile_with_space):
        ffmpeg.ffprobe(test_infile_with_space)
