import contextlib
import os
import shutil

import pytest

from core import commands
import util.youtube as yt

os.environ[yt.YDL_EXECUTE_MODE] = yt.YDL_TESTING_MODE


@contextlib.contextmanager
def temp_dir(path):
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path)


@pytest.fixture
def data_dir():
    return 'tests/.temp/data'


@pytest.fixture
def segments():
    return 'tests/data/segments/test.csv'


@ pytest.fixture
def ontology():
    return 'tests/data/ontology/ontology.json'


@ pytest.fixture
def blacklist():
    return 'tests/data/dataset/blacklist.csv'


def test_download(data_dir, segments, ontology):
    with temp_dir(data_dir):
        commands.dataset.download(('animal',), data_dir, segments, ontology, seed=1)
        success = len(os.listdir(os.path.join(data_dir, 'raw'))) == 3
    assert success


def test_download_with_unavailable_segment(data_dir, segments, ontology):
    with temp_dir(data_dir):
        commands.dataset.download(('human-sounds',), data_dir, segments, ontology, seed=1)
        success = len(os.listdir(os.path.join(data_dir, 'raw'))) == 1
    assert success


def test_download_with_limit(data_dir, segments, ontology):
    with temp_dir(data_dir):
        commands.dataset.download(('animal', 'human-sounds'), data_dir, segments, ontology, limit=1, seed=1)
        success = len(os.listdir(os.path.join(data_dir, 'raw'))) == 2
    assert success


def test_download_with_min_size(data_dir, segments, ontology):
    with temp_dir(data_dir):
        commands.dataset.download(('animal',), data_dir, segments, ontology, min_size=10, seed=1)
        success = len(os.listdir(os.path.join(data_dir, 'raw'))) == 1
    assert success


def test_download_with_max_size(data_dir, segments, ontology):
    with temp_dir(data_dir):
        commands.dataset.download(('animal',), data_dir, segments, ontology, max_size=5, seed=1)
        success = len(os.listdir(os.path.join(data_dir, 'raw'))) == 1
    assert success


def test_download_with_blacklist(data_dir, segments, ontology, blacklist):
    with temp_dir(data_dir):
        commands.dataset.download(('animal',), data_dir, segments, ontology, blacklist=blacklist, seed=1)
        success = len(os.listdir(os.path.join(data_dir, 'raw'))) == 2
    assert success
