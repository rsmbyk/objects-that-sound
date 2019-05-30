import logging
import os
import sys

import youtube_dl
from furl import furl

url_template = 'https://youtube.com/watch'
outfile_extensions = ('mkv', 'mp4', 'webm')

# non-network keyword errors
err_keywords = ['copyright', 'unavailable', 'account', 'Terms of Service', 'terminated', 'removed', 'token',
                'This video is not available.',
                'It is not available in your country.',
                'The uploader has not made this video available in your country.',
                'This video is no longer available due to a privacy claim by a third party.',
                'Watch this video on YouTube. Playback on other websites has been disabled by the video owner.']

logger = logging.Logger('youtube_logger')
logger.addHandler(logging.StreamHandler(sys.stdout))

YDL_EXECUTE_MODE = 'YDL_EXECUTE_MODE'
YDL_NORMAL_MODE = '0'
YDL_TESTING_MODE = '-1'


def dl(v, raise_exception=False, **options):
    """
    Open
    `YoutubeDL Options <https://github.com/rg3/youtube-dl/blob/master/youtube_dl/YoutubeDL.py#L118-L320>`_
    to see all available options
    """
    if os.environ.get(YDL_EXECUTE_MODE, YDL_NORMAL_MODE) == YDL_TESTING_MODE:
        options['logger'] = logger

    ydl = youtube_dl.YoutubeDL(options)

    while True:
        try:
            # youtube video url: https://www.youtube.com/watch?v=[video_id]
            yt = furl(url_template).add(dict(v=v))
            return ydl.download([yt.url])
        except youtube_dl.DownloadError as e:
            if any(map(lambda k: k in str(e), err_keywords)):
                if raise_exception:
                    raise e
                else:
                    return -1


def info(v, raise_exception=False, **options):
    """
    Open
    `YoutubeDL Options <https://github.com/rg3/youtube-dl/blob/master/youtube_dl/YoutubeDL.py#L118-L320>`_
    to see all available options
    """
    if os.environ.get(YDL_EXECUTE_MODE, YDL_NORMAL_MODE) == YDL_TESTING_MODE:
        options['logger'] = logger

    ydl = youtube_dl.YoutubeDL(options)

    while True:
        try:
            # youtube video url: https://www.youtube.com/watch?v=[video_id]
            yt = furl(url_template).add(dict(v=v))
            return ydl.extract_info(yt.url, download=False)
        except youtube_dl.DownloadError as e:
            if any(map(lambda k: k in str(e), err_keywords)):
                if raise_exception:
                    raise e
                else:
                    return -1
