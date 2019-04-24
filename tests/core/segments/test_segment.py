import random

import pytest

from core.segments import Segment
from tests.utils import *
from util import youtube as yt


@pytest.fixture
def test_raw_file():
    return 'tests/data/segments/0qZ3tI4nAZE.mp4'


@pytest.fixture
def root_dir():
    return 'tests/.temp/segments'


@pytest.fixture
def segment():
    return Segment(root_dir='tests/.temp/segments',
                   ytid='0qZ3tI4nAZE',
                   start_seconds=6.000,
                   end_seconds=16.000,
                   positive_labels=['/m/07qrkrw', '/m/09x0r'])


@pytest.fixture
def segment_dict():
    return dict(ytid='0qZ3tI4nAZE',
                start_seconds=6.000,
                end_seconds=16.000,
                positive_labels=['/m/07qrkrw', '/m/09x0r'])


def test_properties(root_dir, segment_dict):
    segment = Segment(root_dir, **segment_dict)
    assert segment.root_dir == root_dir
    assert segment.ytid == segment_dict['ytid']
    assert segment.start_seconds == segment_dict['start_seconds']
    assert segment.end_seconds == segment_dict['end_seconds']
    assert segment.positive_labels == segment_dict['positive_labels']


def test_dir_should_be_inside_root_dir(segment, root_dir):
    cp = os.path.commonprefix([segment.dir, root_dir])
    assert cp == root_dir


def test_frames_dir_should_not_be_in_root_segment_dir(segment):
    assert segment.frames_dir != segment.dir


def test_frames_dir_should_be_inside_segment_dir(segment):
    cp = os.path.commonprefix([segment.frames_dir, segment.dir])
    assert cp == segment.dir


def test_spectrograms_dir_should_not_be_in_root_segment_dir(segment):
    assert segment.spectrograms_dir != segment.dir


def test_spectrograms_dir_should_be_inside_segment_dir(segment):
    cp = os.path.commonprefix([segment.spectrograms_dir, segment.dir])
    assert cp == segment.dir


def test_raw_prefix_should_be_inside_segment_dir(segment):
    cp = os.path.commonprefix([segment.raw_prefix, segment.dir])
    assert cp == segment.dir


def test_raw_prefix_should_not_contains_extension(segment):
    assert '.' not in os.path.basename(segment.raw_prefix)


def test_ydl_outtmpl_should_be_inside_segment_dir(segment):
    cp = os.path.commonprefix([segment.ydl_outtmpl, segment.dir])
    assert cp == segment.dir


def test_ydl_outtmpl_should_contain_extension_placeholder(segment):
    assert segment.ydl_outtmpl.endswith('.%(ext)s')


def test_raw(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir) as raw_file:
            assert segment.raw == raw_file


def test_raw_not_found(segment):
    with pytest.raises(FileNotFoundError):
        assert segment.raw


def test_raw_check_all_extensions(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir) as raw_file:
            for ext in yt.outfile_extensions:
                with temp_move(raw_file, raw_file.replace('mp4', ext)) as out:
                    assert segment.raw == out


def test_raw_should_not_detect_unwanted_extensions(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir) as raw_file:
            with temp_move(raw_file, raw_file.replace('mp4', 'avi')):
                with pytest.raises(FileNotFoundError):
                    assert segment.raw


def test_raw_multiple_files(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir) as raw_file:
            shutil.copy(raw_file, raw_file.replace('mp4', 'mkv'))
            with pytest.raises(AssertionError):
                assert segment.raw


def test_wav_should_be_in_the_same_folder_as_raw(segment):
    assert os.path.dirname(segment.wav) == segment.dir


def test_attrs_file_should_be_in_the_same_directory_as_raw(segment):
    assert os.path.dirname(segment.attrs_file) == segment.dir


def test_frame_should_be_inside_frames_dir(segment):
    cp = os.path.commonprefix([segment.frame(0), segment.frames_dir])
    assert cp == segment.frames_dir


def test_spectrogram_should_be_inside_frames_dir(segment):
    cp = os.path.commonprefix([segment.spectrogram(0), segment.spectrograms_dir])
    assert cp == segment.spectrograms_dir


def test_duration(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert segment.duration == 15.929


def test_frame_rate(segment):
    assert segment.frame_rate == 25


def test_sample_rate(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert segment.sample_rate == 48000


def test_start_frames(segment):
    assert segment.start_frames == segment.start_seconds * segment.frame_rate


def test_end_frames(segment):
    assert segment.end_frames == segment.end_seconds * segment.frame_rate


def test_start_samples(segment):
    assert segment.start_samples == segment.start_seconds * segment.sample_rate


def test_end_samples(segment):
    assert segment.end_samples == segment.end_seconds * segment.sample_rate


def test_get_seconds(segment):
    index = random.randint(segment.start_frames, segment.end_frames)
    assert segment.get_seconds(index) == index / segment.frame_rate


def test_get_sample_index(segment):
    index = random.randint(segment.start_frames, segment.end_frames)
    assert segment.get_sample_index(index) == int(index / segment.frame_rate * segment.sample_rate)


def test_is_available(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert segment.is_available


def test_is_available_fail(segment):
    with temp_dir(segment.dir):
        assert not segment.is_available


def test_len_should_add_padding_at_end(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert len(segment) < int(segment.duration) * segment.frame_rate


def test_wavelength(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert segment.wavelength == 764587


def test_waveform(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            waveform = segment.waveform
            assert waveform.sample_rate == segment.sample_rate
            assert len(waveform.audio) == segment.wavelength


def test_load_frame(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert segment.load_frame(0).shape == (256, 341, 3)
            assert segment.load_frame(len(segment) - 1).shape == (256, 341, 3)


def test_load_spectrogram(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert segment.load_spectrogram(0).shape == (257, 199, 1)
            assert segment.load_spectrogram(len(segment) - 1).shape == (257, 199, 1)


def test_positive_indices_should_be_in_between_start_and_end_seconds(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert segment.start_frames == min(segment.positive_indices)
            assert segment.end_frames >= max(segment.positive_indices)


def test_positive_indices_should_contains_all_indices(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert len(segment.positive_indices) <= 250


def test_positive_indices_should_not_contains_duplicates(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert len(set(segment.positive_indices)) == len(segment.positive_indices)


def test_negative_indices_should_be_outside_start_and_end_seconds(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert segment.start_frames not in segment.negative_indices
            assert segment.end_frames not in segment.negative_indices


def test_negative_indices_should_not_contains_duplicates(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert len(set(segment.negative_indices)) == len(segment.negative_indices)


def test_get_positive_sample_index_should_be_in_1_second_distance(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            for i in range(100):
                frame_index, audio_index = segment.get_positive_sample_index()
                assert abs(frame_index - audio_index) < 25


def test_get_positive_sample_index_should_both_in_positive_ranges(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            for i in range(100):
                frame_index, audio_index = segment.get_positive_sample_index()
                in_frame = segment.start_frames <= frame_index <= segment.end_frames
                in_audio = segment.start_frames - segment.frame_rate <= audio_index <= segment.end_frames
                assert in_frame and in_audio


def test_get_negative_sample_index_should_not_be_in_1_second_distance(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            for i in range(100):
                frame_index, audio_index = segment.get_negative_sample_index()
                assert abs(frame_index - audio_index) > 25


def test_get_negative_sample_index_should_not_be_in_same_range(test_raw_file, segment):
    in_frame_out_audio = False
    out_frame_in_audio = False
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            for i in range(100):
                frame_index, audio_index = segment.get_negative_sample_index()
                in_frame = segment.start_frames <= frame_index <= segment.end_frames
                out_frame = frame_index < segment.start_frames or segment.end_frames > frame_index
                in_audio = segment.start_frames <= audio_index <= segment.end_frames
                out_audio = audio_index < segment.start_frames or segment.end_frames > audio_index
                in_frame_out_audio |= (in_frame and out_audio)
                out_frame_in_audio |= (out_frame and in_audio)
                assert (in_frame and out_audio) or (out_frame and in_audio)
    # assert both type of negative sample generated at least once
    assert in_frame_out_audio and out_frame_in_audio


def test_get_positive_sample(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            frame, spectrogram = segment.get_positive_sample()
            assert frame.shape == (256, 341, 3)
            assert spectrogram.shape == (257, 199, 1)


def test_get_negative_sample(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            frame, spectrogram = segment.get_negative_sample()
            assert frame.shape == (256, 341, 3)
            assert spectrogram.shape == (257, 199, 1)


def test_equality(test_raw_file, segment, segment_dict):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert segment == Segment(segment.root_dir, **segment_dict)


def test_equality_against_str(segment):
    assert segment == '0qZ3tI4nAZE'


def test_equality_fail(segment):
    assert segment != 'failed'


def test_equality_against_invalid_type(segment):
    assert segment != 0
