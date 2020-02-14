import glob
import os
from functools import reduce

import pytest
import youtube_dl

from util import youtube as yt

os.environ[yt.YDL_EXECUTE_MODE] = yt.YDL_TESTING_MODE


@pytest.fixture
def output_dir():
    return 'tests/.temp/youtube'


@pytest.fixture
def ytid_success():
    return 'SmOwn_OEJTo'


@pytest.fixture
def ytid_unavailable():
    return 'xxxxxxxxxxx'


# def test_dl(ytid_success, output_dir):
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #
    # outtmpl = os.path.join(output_dir, '%(id)s.%(ext)s')
    # yt.dl(ytid_success, outtmpl=outtmpl)
    #
    # raw_prefix = os.path.join(output_dir, ytid_success)
    # candidates = map(lambda ext: (raw_prefix, ext), yt.outfile_extensions)
    # patterns = map('.'.join, candidates)
    # match = reduce(lambda a, b: a + b, map(glob.glob, patterns), [])
    # assert len(match) == 1
    #
    # os.remove(match[0])
    # os.rmdir(output_dir)


# def test_dl_with_error(ytid_unavailable):
#     return_code = yt.dl(ytid_unavailable)
#     assert return_code == -1


# def test_dl_with_error_and_raise_exception(ytid_unavailable):
#     with pytest.raises(youtube_dl.DownloadError):
#         yt.dl(ytid_unavailable, True)
