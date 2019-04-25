from copy import copy

import math
import pytest

from core.generator import SegmentsGenerator
from core.models import AVOLNet
from core.segments import Segment
from tests.utils import *


def map_dir(segments):
    return list(map(lambda s: s.dir, segments))


def map_raw(segments):
    return list(map(lambda s: '{}.mp4'.format(os.path.join(s.dir, s.ytid)), segments))


@pytest.fixture
def test_raw_file():
    return 'tests/data/segments/0qZ3tI4nAZE.mp4'


@pytest.fixture
def segment():
    with temp_dir('tests/.temp/segments/0qZ3tI4nAZE'):
        return Segment(root_dir='tests/.temp/segments',
                       ytid='0qZ3tI4nAZE',
                       start_seconds=6.000,
                       end_seconds=16.000,
                       positive_labels=['/m/07qrkrw', '/m/09x0r'])


@pytest.fixture
def segments(segment):
    return 100 * [segment]


@pytest.fixture
def model():
    return AVOLNet()


def test_with_invalid_segments_type():
    with pytest.raises(TypeError):
        assert SegmentsGenerator(0, None)


def test_with_invalid_segments_item_type(model):
    with pytest.raises(TypeError):
        assert SegmentsGenerator([0], model)


def test_with_single_segment(test_raw_file, segment, model):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert len(SegmentsGenerator(segment, model).segments) == 1


def test_with_empty_segments(model):
    with pytest.raises(ValueError):
        assert SegmentsGenerator([], model)


def test_with_unavailable_segments(segments, model):
    with pytest.raises(ValueError):
        assert SegmentsGenerator(segments, model)


def test_batch_size_default(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model)
            assert generator.batch_size == 16
            generator = SegmentsGenerator(segments, model, None)
            assert generator.batch_size == 16


def test_batch_size_invalid_type(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            with pytest.raises(TypeError):
                assert SegmentsGenerator(segments, model, '32')


def test_batch_size_invalid_value(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            with pytest.raises(ValueError):
                assert SegmentsGenerator(segments, model, -16)
            with pytest.raises(ValueError):
                assert SegmentsGenerator(segments, model, 0)


def test_len(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 25)
            assert len(generator) == math.ceil(len(segments) / 25)


def test_len_should_ceil_the_batch_count(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 16)
            assert len(generator) == math.ceil(len(segments) / 16)


def test_model_with_invalid_type(test_raw_file, segments):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            with pytest.raises(TypeError):
                assert SegmentsGenerator(segments, 0)


def test_getitem(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 16)
            batch = generator[0]
            zipped = list(zip(batch[0][0], batch[0][1], batch[1]))
            assert len(zipped) == 32


def test_getitem_with_unavailable_negative_segment(test_raw_file, segments, model):
    cp = copy(segments[0])
    cp.start_seconds = 0
    cp.end_seconds = 20
    segments[0] = cp
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 16)
            batch = generator[0]
            zipped = list(zip(batch[0][0], batch[0][1], batch[1]))
            assert len(zipped) == 31
