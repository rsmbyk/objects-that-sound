import os

import pytest

from util import filesystem as fs


@pytest.fixture
def test_source_dir():
    return 'tests/data/filesystem/source'


@pytest.fixture
def non_existing_test_source_dir():
    return 'tests/.temp/filesystem/not_source'


@pytest.fixture
def test_target_dir():
    return 'tests/.temp/filesystem/target'


@pytest.fixture
def non_existing_test_target_dir():
    return 'tests/.temp/filesystem/not_target'


def test_symlink(test_source_dir, test_target_dir):
    link_name = fs.symlink(test_source_dir, test_target_dir)
    assert os.path.exists(link_name)
    os.remove(link_name)
    os.rmdir(test_target_dir)


def test_symlink_with_alias(test_source_dir, test_target_dir):
    link_name = fs.symlink(test_source_dir, test_target_dir, 'alias')
    assert os.path.exists(link_name)
    os.remove(link_name)
    os.rmdir(test_target_dir)


def test_symlink_with_non_existing_source_dir(non_existing_test_source_dir):
    with pytest.raises(FileNotFoundError):
        fs.symlink(non_existing_test_source_dir, None)


def test_symlink_should_create_target_directory(test_source_dir, non_existing_test_target_dir):
    if os.path.exists(non_existing_test_target_dir):
        os.rmdir(non_existing_test_target_dir)
    link_name = fs.symlink(test_source_dir, non_existing_test_target_dir)
    assert os.path.exists(non_existing_test_target_dir)
    os.remove(link_name)
    os.rmdir(non_existing_test_target_dir)
