import os
import shutil

import pytest

from core import ops


@pytest.fixture
def test_video_file():
    return 'tests/data/ops/test.mkv'


@pytest.fixture
def non_existing_test_video_file():
    return 'tests/data/ops/missing.mkv'


@pytest.fixture
def output_dir():
    return 'tests/.temp/ops/frames'


def test_extract_frames(test_video_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ops.extract_frames(test_video_file, output_dir)
    assert os.path.exists(output_dir)
    assert len(os.listdir(output_dir)) == 184
    shutil.rmtree(os.path.dirname(output_dir))


def test_extract_frames_with_non_existing_video(non_existing_test_video_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        ops.extract_frames(non_existing_test_video_file, output_dir)
    os.removedirs(output_dir)


def test_extract_frames_with_non_dir_output_dir(test_video_file, non_existing_test_video_file):
    with open(non_existing_test_video_file, 'a'):
        with pytest.raises(NotADirectoryError):
            ops.extract_frames(test_video_file, non_existing_test_video_file)
    os.remove(non_existing_test_video_file)


def test_extract_frames_should_create_output_parent_dir(test_video_file, output_dir):
    ops.extract_frames(test_video_file, output_dir)
    assert os.path.exists(output_dir)
    shutil.rmtree(os.path.dirname(output_dir))


def test_extract_frames_with_start_time_offset(test_video_file, output_dir):
    ops.extract_frames(test_video_file, output_dir, start_time=3)
    assert 100 <= len(os.listdir(output_dir)) < 125
    shutil.rmtree(os.path.dirname(output_dir))


def test_extract_frames_with_stop_time_offset(test_video_file, output_dir):
    ops.extract_frames(test_video_file, output_dir, stop_time=4)
    assert len(os.listdir(output_dir)) == 100
    shutil.rmtree(os.path.dirname(output_dir))


def test_extract_frames_with_start_and_stop_time_offset(test_video_file, output_dir):
    ops.extract_frames(test_video_file, output_dir, start_time=3, stop_time=4)
    assert len(os.listdir(output_dir)) == 25
    shutil.rmtree(os.path.dirname(output_dir))
