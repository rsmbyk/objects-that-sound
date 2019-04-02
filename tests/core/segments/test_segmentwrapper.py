import pytest

from core.segments import SegmentsWrapper


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
    return SegmentsWrapper('tests/data/segments/test.csv',
                           'tests/.temp/segments')


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


def test_load_segments_with_lot_of_segments(test_segments_file_with_lot_of_comments):
    segments = SegmentsWrapper(test_segments_file_with_lot_of_comments, None)
    assert len(segments) == 25


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
