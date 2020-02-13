import contextlib
import os
import shutil

import pytest

from core import commands
from core.ontology import Ontology
from core.segments import SegmentsWrapper


@contextlib.contextmanager
def temp_dir(path):
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path)


@pytest.fixture
def data_dir():
    return 'tests/.temp/preprocess/data'


@pytest.fixture
def segments(data_dir):
    return SegmentsWrapper('tests/data/segments/test.csv', os.path.join(data_dir, 'raw'))


@pytest.fixture
def ontology(data_dir):
    return Ontology('tests/data/ontology/ontology.json', os.path.join(data_dir, 'videos'))


def test_compress_segments(data_dir, segments, ontology):
    with temp_dir(data_dir):
        outfile = os.path.join(data_dir, 'compressed_segments.csv')
        commands.dataset.download(('animal',), data_dir, segments.filename, ontology.filename, limit=1, max_size=10)
        commands.dataset.compress_segments(data_dir, segments.filename, ontology.filename, ('animal',), outfile)
        segments = SegmentsWrapper(outfile, os.path.join(data_dir, 'raw'))
        segments_length = len(segments)
    assert segments_length == 1
