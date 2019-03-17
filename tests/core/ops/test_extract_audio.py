import os

import pytest

from core import ops


@pytest.fixture
def test_video_file():
    return 'tests/data/ops/test.mkv'


@pytest.fixture
def non_existing_test_video_file():
    return 'tests/data/ops/missing.mkv'


@pytest.fixture
def output():
    return 'tests/.temp/ops/audio.wav'


@pytest.fixture
def output_with_missing_parent_dir():
    return 'tests/.temp/ops/missing/audio.wav'


def test_extract_audio(test_video_file, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    ops.extract_audio(test_video_file, output)
    assert os.path.exists(output)
    os.remove(output)
    os.removedirs(os.path.dirname(output))


def test_extract_audio_with_non_existing_video(non_existing_test_video_file, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with pytest.raises(FileNotFoundError):
        ops.extract_audio(non_existing_test_video_file, output)
    os.removedirs(os.path.dirname(output))


def test_extract_audio_should_skip_already_extracted_audio(test_video_file, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    ops.extract_audio(test_video_file, output)
    created_time = os.path.getmtime(output)
    ops.extract_audio(test_video_file, output)
    assert os.path.getmtime(output) == created_time
    os.remove(output)
    os.removedirs(os.path.dirname(output))


def test_extract_audio_should_create_output_parent_dir(test_video_file, output_with_missing_parent_dir):
    ops.extract_audio(test_video_file, output_with_missing_parent_dir)
    assert os.path.exists(os.path.dirname(output_with_missing_parent_dir))
    os.remove(output_with_missing_parent_dir)
    os.removedirs(os.path.dirname(output_with_missing_parent_dir))
