import contextlib
import os
import shutil

import pytest

from core import commands
from core.ontology import Ontology
from core.segments import SegmentsWrapper
from util import youtube as yt


@contextlib.contextmanager
def temp_data_dir(segments):
    s = segments[0]
    yt.dl(s.ytid, outtmpl=s.ydl_outtmpl)
    yield s
    shutil.rmtree(os.path.dirname(s.root_dir))


@pytest.fixture
def data_dir():
    return 'tests/.temp/preprocess/data'


@pytest.fixture
def segments(data_dir):
    return SegmentsWrapper('tests/data/segments/test.csv', os.path.join(data_dir, 'raw'))


@pytest.fixture
def ontology(data_dir):
    return Ontology('tests/data/ontology/ontology.json', os.path.join(data_dir, 'videos'))


@pytest.fixture
def blacklist():
    return 'tests/data/dataset/blacklist.csv'


def test_preprocess(data_dir, segments):
    with temp_data_dir(segments) as s:
        commands.dataset.preprocess(data_dir, segments.filename)
        assert os.path.exists(s.frame(s.start_frames))
        assert os.path.exists(s.frame(s.end_frames - 1))
        assert os.path.exists(s.spectrogram(s.start_frames))
        assert os.path.exists(s.spectrogram(s.end_frames - 1))


def test_preprocess_with_workers(data_dir, segments):
    with temp_data_dir(segments) as s:
        commands.dataset.preprocess(data_dir, segments.filename, workers=4)
        assert os.path.exists(s.frame(s.start_frames))
        assert os.path.exists(s.frame(s.end_frames - 1))
        assert os.path.exists(s.spectrogram(s.start_frames))
        assert os.path.exists(s.spectrogram(s.end_frames - 1))


def test_preprocess_with_invalid_type_of_workers(data_dir, segments):
    with temp_data_dir(segments):
        with pytest.raises(TypeError):
            commands.dataset.preprocess(data_dir, segments.filename, workers='4')


def test_preprocess_with_invalid_value_of_workers(data_dir, segments):
    with temp_data_dir(segments):
        with pytest.raises(ValueError):
            commands.dataset.preprocess(data_dir, segments.filename, workers=-1)


def test_preprocess_should_process_available_segments_only(data_dir, segments):
    with temp_data_dir(segments):
        commands.dataset.preprocess(data_dir, segments.filename)
        assert not os.path.exists(segments[1].frame(segments[1].start_frames))
        assert not os.path.exists(segments[1].frame(segments[1].end_frames - 1))
        assert not os.path.exists(segments[1].spectrogram(segments[1].start_frames))
        assert not os.path.exists(segments[1].spectrogram(segments[1].end_frames - 1))
