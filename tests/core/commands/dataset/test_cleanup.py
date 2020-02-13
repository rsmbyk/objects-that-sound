import contextlib
import os
import shutil

import pytest

from core import commands
from core.segments import SegmentsWrapper
from util import youtube as yt

os.environ[yt.YDL_EXECUTE_MODE] = yt.YDL_TESTING_MODE


@contextlib.contextmanager
def temp_data_dir(segments):
    s = segments[2]
    yt.dl(s.ytid, outtmpl=s.ydl_outtmpl)
    yield s
    shutil.rmtree(os.path.dirname(s.root_dir))


@pytest.fixture
def data_dir():
    return 'tests/.temp/data'


@pytest.fixture
def segments(data_dir):
    return SegmentsWrapper('tests/data/segments/test.csv', os.path.join(data_dir, 'raw'))


def test_cleanup_all(data_dir, segments):
    with temp_data_dir(segments) as s:
        commands.dataset.preprocess(data_dir, segments.filename)
        commands.dataset.cleanup(data_dir, segments.filename, audio=True, frames=True, spectrograms=True)
        audio_removed = not os.path.exists(s.wav)
        frames_removed = not os.path.exists(s.frames_dir)
        spectrograms_removed = not os.path.exists(s.spectrograms_dir)
    assert audio_removed and frames_removed and spectrograms_removed


def test_cleanup_audio_only(data_dir, segments):
    with temp_data_dir(segments) as s:
        commands.dataset.preprocess(data_dir, segments.filename)
        commands.dataset.cleanup(data_dir, segments.filename, audio=True, frames=False, spectrograms=False)
        audio_removed = not os.path.exists(s.wav)
        frames_removed = not os.path.exists(s.frames_dir)
        spectrograms_removed = not os.path.exists(s.spectrograms_dir)
    assert audio_removed and not frames_removed and not spectrograms_removed


def test_cleanup_frames_only(data_dir, segments):
    with temp_data_dir(segments) as s:
        commands.dataset.preprocess(data_dir, segments.filename)
        commands.dataset.cleanup(data_dir, segments.filename, audio=False, frames=True, spectrograms=False)
        audio_removed = not os.path.exists(s.wav)
        frames_removed = not os.path.exists(s.frames_dir)
        spectrograms_removed = not os.path.exists(s.spectrograms_dir)
    assert not audio_removed and frames_removed and not spectrograms_removed


def test_cleanup_spectrograms_only(data_dir, segments):
    with temp_data_dir(segments) as s:
        commands.dataset.preprocess(data_dir, segments.filename)
        commands.dataset.cleanup(data_dir, segments.filename, audio=False, frames=False, spectrograms=True)
        audio_removed = not os.path.exists(s.wav)
        frames_removed = not os.path.exists(s.frames_dir)
        spectrograms_removed = not os.path.exists(s.spectrograms_dir)
    assert not audio_removed and not frames_removed and spectrograms_removed


def test_cleanup_no_remove(data_dir, segments):
    with temp_data_dir(segments) as s:
        commands.dataset.preprocess(data_dir, segments.filename)
        commands.dataset.cleanup(data_dir, segments.filename, audio=False, frames=False, spectrograms=False)
        audio_removed = not os.path.exists(s.wav)
        frames_removed = not os.path.exists(s.frames_dir)
        spectrograms_removed = not os.path.exists(s.spectrograms_dir)
    assert not audio_removed and not frames_removed and not spectrograms_removed


def test_cleanup_empty_segment(data_dir, segments):
    with temp_data_dir(segments) as s:
        commands.dataset.cleanup(data_dir, segments.filename, audio=False, frames=False, spectrograms=False)
        audio_removed = not os.path.exists(s.wav)
        frames_removed = not os.path.exists(s.frames_dir)
        spectrograms_removed = not os.path.exists(s.spectrograms_dir)
    assert audio_removed and frames_removed and spectrograms_removed
