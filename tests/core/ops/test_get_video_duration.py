import pytest

from core import ops


@pytest.fixture
def test_video_file():
    return 'tests/data/ops/test.mkv'


@pytest.fixture
def non_existing_test_video_file():
    return 'tests/data/ops/missing.mkv'


def test_get_video_duration(test_video_file):
    duration = ops.get_video_duration(test_video_file)
    assert duration == 7.421


def test_get_non_existing_video_duration(non_existing_test_video_file):
    with pytest.raises(FileNotFoundError):
        ops.get_video_duration(non_existing_test_video_file)
