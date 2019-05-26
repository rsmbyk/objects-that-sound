import json
import random

import pytest

from core.segments import Segment
from tests.utils import *
from util import youtube as yt


@pytest.fixture
def test_raw_file():
    return 'tests/data/segments/0qZ3tI4nAZE.mp4'


@pytest.fixture
def test_attributes_file():
    return 'tests/data/segments/0qZ3tI4nAZE.json'


@pytest.fixture
def test_incomplete_attributes_file():
    return 'tests/data/segments/incomplete.json'


@pytest.fixture
def test_invalid_attributes_file():
    return 'tests/data/segments/invalid.json'


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


def test_pre_load_attributes_file(root_dir, test_attributes_file, segment_dict):
    with temp_dir(os.path.join(root_dir, segment_dict['ytid'])):
        with temp_copy(test_attributes_file, os.path.join(root_dir, segment_dict['ytid'])):
            segment = Segment(root_dir, **segment_dict)
            assert segment.duration == 15.929
            assert segment.wavelength == 764587


def test_load_attributes_should_overwrite_current_values(test_attributes_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_attributes_file, segment.dir):
            segment._Segment__duration = 0
            segment._Segment__wavelength = 0
            segment.load_attributes()
            assert segment.duration == 15.929
            assert segment.wavelength == 764587


def test_load_attributes_with_invalid_content(test_invalid_attributes_file, segment):
    segment._Segment__duration = 0
    with temp_dir(segment.dir):
        with temp_copy_file(test_invalid_attributes_file, segment.attrs_file):
            segment.load_attributes()
            assert segment.duration == 0


def test_save_attribute(test_incomplete_attributes_file, segment):
    with temp_dir(segment.dir):
        with temp_copy_file(test_incomplete_attributes_file, segment.attrs_file):
            segment.save_attribute('duration', 20)

            with open(segment.attrs_file) as attrs_file:
                attrs = json.load(attrs_file)
                assert attrs['duration'] == 20


def test_save_attribute_should_create_file_if_not_exists(segment):
    with temp_dir(segment.dir):
        assert not os.path.exists(segment.attrs_file)
        segment.save_attribute('duration', 20)
        assert os.path.exists(segment.attrs_file)


def test_save_attribute_should_not_create_file_if_segment_dir_does_not_exists(segment):
    segment.save_attribute('duration', 20)
    assert not os.path.exists(segment.attrs_file)


def test_save_attribute_with_invalid_key(segment):
    with pytest.raises(ValueError):
        segment.save_attribute('invalid_key', 20)


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
            assert waveform.sample_rate.numpy() == segment.sample_rate
            assert len(waveform.audio) == segment.wavelength


def test_load_frame(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert segment.load_frame(segment.start_frames).shape == (256, 341, 3)
            assert segment.load_frame(len(segment) - 1).shape == (256, 341, 3)


def test_load_spectrogram(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert segment.load_spectrogram(segment.start_frames).shape == (257, 199, 1)
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


def test_get_positive_sample(test_raw_file, segment):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            frame, spectrogram = segment.get_positive_sample()
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
