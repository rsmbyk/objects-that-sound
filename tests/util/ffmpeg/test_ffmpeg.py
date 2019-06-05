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


@contextlib.contextmanager
def temp_dir(path):
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path)


@pytest.fixture
def test_infile():
    return 'tests/data/ffmpeg/test.mp4'


@pytest.fixture
def non_existing_test_infile():
    return 'tests/data/ffmpeg/infile'


@pytest.fixture
def test_outfile():
    return 'tests/.temp/ffmpeg/out/file.mp4'


@pytest.fixture
def test_infile_with_space():
    return 'tests/.temp/ffmpeg/input file/test.mp4'


@pytest.fixture
def test_outfile_with_space():
    return 'tests/.temp/ffmpeg/out dir/file.mp4'


def test_ffmpeg_not_available():
    ffmpeg_path = shutil.which('ffmpeg')

    if ffmpeg_path:
        # remove ffmpeg from PATH
        path = os.environ['PATH'].split(';')
        path.remove(os.path.dirname(ffmpeg_path))
        os.environ['PATH'] = ';'.join(path)

    assert not shutil.which('ffmpeg')

    with pytest.raises(AssertionError):
        ffmpeg.ffmpeg(None, None)

    if ffmpeg_path:
        # put ffmpeg back to PATH
        path = os.environ['PATH'].split(';')
        path.append(os.path.dirname(ffmpeg_path))
        os.environ['PATH'] = ';'.join(path)

    assert shutil.which('ffmpeg')


def test_ffmpeg_with_non_existing_infile(non_existing_test_infile):
    with pytest.raises(FileNotFoundError):
        ffmpeg.ffmpeg(non_existing_test_infile, None)


def test_ffmpeg_should_create_outfile_directory(test_infile, test_outfile):
    if os.path.exists(os.path.dirname(test_outfile)):
        os.rmdir(os.path.dirname(test_outfile))
    ffmpeg.ffmpeg(test_infile, test_outfile)
    assert os.path.exists(os.path.dirname(test_outfile))
    os.remove(test_outfile)
    os.rmdir(os.path.dirname(test_outfile))


def test_ffmpeg_process_video(test_infile, test_outfile):
    ffmpeg.ffmpeg(test_infile, test_outfile)
    assert os.path.exists(test_outfile)
    os.remove(test_outfile)
    os.rmdir(os.path.dirname(test_outfile))


def test_ffmpeg_with_invalid_parameters(test_infile, test_outfile):
    with pytest.raises(RuntimeError):
        ffmpeg.ffmpeg(test_infile, test_outfile, 'invalid_flag')
    with pytest.raises(RuntimeError):
        ffmpeg.ffmpeg(test_infile, test_outfile, invalid_option=True)
    os.rmdir(os.path.dirname(test_outfile))


def test_ffmpeg_with_path_containing_space(test_infile, test_infile_with_space, test_outfile_with_space):
    with temp_copy(test_infile, test_infile_with_space):
        with temp_dir(os.path.dirname(test_outfile_with_space)):
            ffmpeg.ffmpeg(test_infile_with_space, test_outfile_with_space)
            assert os.path.exists(test_outfile_with_space)
