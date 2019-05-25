import youtube_dl
from furl import furl

url_template = 'https://youtube.com/watch'
outfile_extensions = ('mkv', 'mp4', 'webm')

# non-network keyword errors
err_keywords = ['copyright', 'unavailable', 'account', 'Terms of Service', 'terminated', 'removed', 'token',
                'It is not available in your country.',
                'The uploader has not made this video available in your country.']


def dl(v, raise_exception=False, **options):
    """
    Open
    `YoutubeDL Options <https://github.com/rg3/youtube-dl/blob/master/youtube_dl/YoutubeDL.py#L118-L320>`_
    to see all available options
    """

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
