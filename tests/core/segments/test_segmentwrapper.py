import os
import shutil
import time
from contextlib import contextmanager

import pandas as pd
import pytest

from core.segments import SegmentsWrapper


@contextmanager
def tempdir(path):
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path)


@contextmanager
def copy_raw_file(to):
    to = os.path.join(to, os.path.basename(to) + '.mp4')
    yield shutil.copyfile('tests/data/segments/0qZ3tI4nAZE.mp4', to)
    os.remove(to)


@pytest.fixture
def test_segments_file():
    return 'tests/data/segments/test.csv'


@pytest.fixture
def test_segments_file_with_lot_of_comments():
    return 'tests/data/segments/lot_of_comments.csv'


@pytest.fixture
def test_non_existing_segments_file():
    return 'tests/data/segments/missing.csv'


@pytest.fixture
def segments():
    s = pd.read_csv('tests/data/segments/test.csv',
                    sep=', ',
                    header=None,
                    engine='python')

    def multi_tempdir(i=0):
        if i < len(s):
            with tempdir(os.path.join('tests/.temp/segments', s[0][i])):
                with copy_raw_file(os.path.join('tests/.temp/segments', s[0][i])):
                    return multi_tempdir(i + 1)
        else:
            segments_wrapper = SegmentsWrapper('tests/data/segments/test.csv',
                                               'tests/.temp/segments')
            time.sleep(20)
            assert segments_wrapper.segments is not None
            return segments_wrapper

    return multi_tempdir()


@pytest.fixture
def segments_with_lot_of_comments():
    s = pd.read_csv('tests/data/segments/test.csv',
                    sep=', ',
                    header=None,
                    engine='python')

    def multi_tempdir(i=0):
        if i < len(s):
            with tempdir(os.path.join('tests/.temp/segments', s[0][i])):
                with copy_raw_file(os.path.join('tests/.temp/segments', s[0][i])):
                    return multi_tempdir(i + 1)
        else:
            segments_wrapper = SegmentsWrapper('tests/data/segments/lot_of_comments.csv',
                                               'tests/.temp/segments')
            time.sleep(20)
            assert segments_wrapper.segments is not None
            return segments_wrapper

    return multi_tempdir()


def test_create_segment_wrapper(segments, test_segments_file):
    assert segments.filename == test_segments_file


def test_create_with_non_existing_segments_file(test_non_existing_segments_file):
    with pytest.raises(FileNotFoundError):
        assert SegmentsWrapper(test_non_existing_segments_file, None)


def test_create_with_invalid_filename_type():
    with pytest.raises(TypeError):
        assert SegmentsWrapper(0, None)


def test_len(segments):
    assert len(segments) == 25


def test_load_segments_with_lot_of_segments(segments_with_lot_of_comments):
    assert len(segments_with_lot_of_comments) == 25


def test_to_dict(segments):
    assert len(segments.to_dict()) == 25


def test_to_dict_should_contains_all_items(segments):
    segments_dict = segments.to_dict()
    assert all(map(lambda s: s.ytid in segments_dict, segments))


def test_contains(segments):
    assert '--aE2O5G5WE' in segments


def test_contains_missing_key(segments):
    assert 'missing' not in segments


def test_getitem_by_ytid(segments):
    item = segments['--aE2O5G5WE']
    assert item.ytid == '--aE2O5G5WE'
    assert item.start_seconds == 0
    assert item.end_seconds == 10
    assert item.positive_labels == ['/m/03fwl', '/m/04rlf', '/m/09x0r']


def test_getitem_by_index(segments):
    item = segments[0]
    assert item.ytid == '--PJHxphWEs'
    assert item.start_seconds == 30
    assert item.end_seconds == 40
    assert item.positive_labels == ['/m/09x0r', '/t/dd00088']


def test_getitem_by_slice(segments):
    items = segments[:10]
    assert len(items) == 10


def test_getitem_by_invalid_type(segments):
    with pytest.raises(KeyError):
        assert segments[[]]


def test_filter(segments):
    filtered = segments.filter('/m/09x0r')
    assert len(filtered) == 3


def test_filter_by_invalid_type(segments):
    with pytest.raises(TypeError):
        assert segments.filter(0)
