import logging
from io import StringIO

import pytest
import youtube_dl

from util import youtube as yt


class Han(logging.Handler):
    def handle(self, record):
        print(record)


@pytest.fixture
def logger():
    return logging.Logger('test_dl_logger')


@pytest.fixture
def ytid_success():
    return '00mE-lhe_R8'


@pytest.fixture
def ytid_unavailable():
    return '-3IYpJfLVJk'


def test_dl(ytid_success, logger):
    info = yt.info(ytid_success, logger=logger)
    assert type(info) == dict


def test_dl_with_error(ytid_unavailable, logger):
    return_code = yt.info(ytid_unavailable, logger=logger)
    assert return_code == -1


def test_dl_with_error_and_raise_exception(ytid_unavailable, logger):
    with pytest.raises(youtube_dl.DownloadError):
        yt.info(ytid_unavailable, True, logger=logger)
