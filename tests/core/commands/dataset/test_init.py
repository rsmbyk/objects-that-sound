import contextlib
import os
import shutil
import stat
from pathlib import Path

import pytest

from core import commands


@contextlib.contextmanager
def temp_dir(path):
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path)


def iterate_dataset(data_dir, dataset_files):
    for parent_dir, urls in dataset_files.items():
        files = list(map(lambda x: x.split('/')[-1], urls))
        for file in files:
            yield os.path.join(data_dir, parent_dir, file)


@contextlib.contextmanager
def cleanup_dataset_files(data_dir, dataset_files):
    yield data_dir, dataset_files
    for filename in iterate_dataset(data_dir, dataset_files):
        os.chmod(filename, stat.S_IWUSR | stat.S_IREAD)


@pytest.fixture
def data_dir():
    return 'tests/.temp/data'


@pytest.fixture
def dataset_files():
    return {
        'assessments': [
            'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/qa/qa_true_counts.csv',
            'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/qa/rerated_video_ids.txt'
        ],
        'labels': [
            'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv',
            'https://raw.githubusercontent.com/audioset/ontology/master/ontology.json'
        ],
        'segments': [
            'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv',
            'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv',
            'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv'
        ]
    }


def test_init_dataset(data_dir, dataset_files):
    with temp_dir(data_dir):
        with cleanup_dataset_files(data_dir, dataset_files):
            commands.dataset.init(data_dir, False)
            assert all(map(os.path.exists, iterate_dataset(data_dir, dataset_files)))


def test_init_exists(data_dir, dataset_files):
    with temp_dir(data_dir):
        with cleanup_dataset_files(data_dir, dataset_files):
            for filename in iterate_dataset(data_dir, dataset_files):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                Path(filename).touch()

            commands.dataset.init(data_dir, False)
            assert all(map(os.path.exists, iterate_dataset(data_dir, dataset_files)))


def test_init_overwrite(data_dir, dataset_files):
    with temp_dir(data_dir):
        with cleanup_dataset_files(data_dir, dataset_files):
            for filename in iterate_dataset(data_dir, dataset_files):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                Path(filename).touch()

            commands.dataset.init(data_dir, True)
            assert all(map(os.path.exists, iterate_dataset(data_dir, dataset_files)))


def test_init_clean_up(data_dir, dataset_files):
    with temp_dir(data_dir):
        with cleanup_dataset_files(data_dir, dataset_files):
            temp_file = os.path.join(data_dir, 'segments', 'temp_file')
            os.makedirs(os.path.dirname(temp_file))
            Path(temp_file).touch()

            commands.dataset.init(data_dir, False)
            assert not os.path.exists(temp_file)
