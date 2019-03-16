import pytest

from util import filereader as fr


@pytest.fixture
def test_file():
    return 'tests/data/filereader/test.txt'


def test_read_txt(test_file):
    content = fr.read_txt(test_file)
    assert len(content) == 5
