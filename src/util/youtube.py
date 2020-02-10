import logging
import os
import sys

import youtube_dl
from furl import furl

url_template = 'https://youtube.com/watch'
outfile_extensions = ('mkv', 'mp4', 'webm')

# non-network keyword errors
err_keywords = ['"token" parameter not in video info for unknown reason;',
                'requested format not available;',
                'This video is private.',
                'This video is unavailable.',
                'This video is not available.',
                'This video has been removed by the user',
                'This video has been removed for violating YouTube\'s Terms of Service.',
                'The uploader has not made this video available in your country.',
                'Watch this video on YouTube. Playback on other websites has been disabled by the video owner.',
                'This video is a duplicate of another YouTube video',

                'The YouTube account associated with this video has been terminated'
                ' due to multiple third-party notifications of copyright infringement.',

                'This video is no longer available',
                ' because the YouTube account associated with this video has been terminated.',
                ' due to a privacy claim by a third party.',
                ' due to a copyright claim by ',
                ' is no longer available due to a copyright claim by a third party.',

                'This video contains content from ',
                '. It is not available in your country.',
                ', who has blocked it on copyright grounds.',
                ', one or more of whom have blocked it on copyright grounds.', ]

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
