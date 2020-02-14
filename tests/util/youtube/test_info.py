import os

import pytest
import youtube_dl

from util import youtube as yt

os.environ[yt.YDL_EXECUTE_MODE] = yt.YDL_TESTING_MODE


@pytest.fixture
def ytid_success():
    return 'SmOwn_OEJTo'


@pytest.fixture
def ytid_unavailable():
    return 'xxxxxxxxxxx'


def test_dl(ytid_success):
    info = yt.info(ytid_success)
    assert type(info) == dict


# def test_dl_with_error(ytid_unavailable):
#     return_code = yt.info(ytid_unavailable)
#     assert return_code == -1


# def test_dl_with_error_and_raise_exception(ytid_unavailable):
#     with pytest.raises(youtube_dl.DownloadError):
#         yt.info(ytid_unavailable, True)
