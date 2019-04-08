import os
import shutil
from contextlib import contextmanager
from copy import copy

import math
import numpy as np
import pytest

from core.generator import SegmentsGenerator
from core.models import AVOLNet
from core.segments import Segment


@contextmanager
def tempdir(segments):
    for segment in segments:
        os.makedirs(segment.dir, exist_ok=True)
    yield len(segments)
    for segment in segments:
        if os.path.exists(segment.dir):
            shutil.rmtree(segment.dir)


@contextmanager
def copy_raw_file(segments):
    files = list()
    for segment in segments:
        files.append(
            shutil.copyfile('tests/data/segments/0qZ3tI4nAZE.mp4', '{}.mp4'.format(segment.raw_prefix)))
    yield files
    for file in files:
        if os.path.exists(file):
            os.remove(file)


@pytest.fixture
def segments():
    return 100 * [Segment(root_dir='tests/.temp/segments',
                          ytid='0qZ3tI4nAZE',
                          start_seconds=6.000,
                          end_seconds=16.000,
                          positive_labels=['/m/07qrkrw', '/m/09x0r'])]


@pytest.fixture
def segment():
    return Segment(root_dir='tests/.temp/segments',
                   ytid='0qZ3tI4nAZE',
                   start_seconds=6.000,
                   end_seconds=16.000,
                   positive_labels=['/m/07qrkrw', '/m/09x0r'])


@pytest.fixture
def model():
    return AVOLNet()


def test_with_invalid_segments_type():
    with pytest.raises(TypeError):
        assert SegmentsGenerator(0, None)


def test_with_invalid_segments_item_type(model):
    with pytest.raises(TypeError):
        assert SegmentsGenerator([0], model)


def test_with_single_segment(segment, model):
    with tempdir([segment]):
        with copy_raw_file([segment]):
            assert len(SegmentsGenerator(segment, model).segments) == 1


def test_with_empty_segments(model):
    with pytest.raises(ValueError):
        assert SegmentsGenerator([], model)


def test_with_unavailable_segments(segments, model):
    with pytest.raises(ValueError):
        assert SegmentsGenerator(segments, model)


def test_batch_size_default(segments, model):
    with tempdir(segments):
        with copy_raw_file(segments):
            generator = SegmentsGenerator(segments, model)
            assert generator.batch_size == 16
            generator = SegmentsGenerator(segments, model, None)
            assert generator.batch_size == 16


def test_batch_size_invalid_type(segments, model):
    with tempdir(segments):
        with copy_raw_file(segments):
            with pytest.raises(TypeError):
                assert SegmentsGenerator(segments, model, '32')


def test_batch_size_invalid_value(segments, model):
    with tempdir(segments):
        with copy_raw_file(segments):
            with pytest.raises(ValueError):
                assert SegmentsGenerator(segments, model, -16)
            with pytest.raises(ValueError):
                assert SegmentsGenerator(segments, model, 0)


def test_len(segments, model):
    with tempdir(segments):
        with copy_raw_file(segments):
            generator = SegmentsGenerator(segments, model, 25)
            assert len(generator) == math.ceil(len(segments) / 25)


def test_len_should_ceil_the_batch_count(segments, model):
    with tempdir(segments):
        with copy_raw_file(segments):
            generator = SegmentsGenerator(segments, model, 16)
            assert len(generator) == math.ceil(len(segments) / 16)


def test_model_with_invalid_type(segments):
    with tempdir(segments):
        with copy_raw_file(segments):
            with pytest.raises(TypeError):
                assert SegmentsGenerator(segments, 0)


def test_getitem(segments, model):
    with tempdir(segments):
        with copy_raw_file(segments):
            generator = SegmentsGenerator(segments, model, 16)
            batch = generator[0]
            zipped = list(zip(batch[0][0], batch[0][1], batch[1]))
            assert len(np.array(zipped)) == 32


def test_getitem_with_unavailable_negative_segment(segments, model):
    segments[0] = copy(segments[0])
    segments[0].start_seconds = 0
    segments[0].end_seconds = 20
    with tempdir(segments):
        with copy_raw_file(segments):
            generator = SegmentsGenerator(segments, model, 16)
            batch = generator[0]
            zipped = list(zip(batch[0][0], batch[0][1], batch[1]))
            assert len(np.array(zipped)) == 31
