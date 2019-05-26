import math
import numpy as np
import pytest

from core.augmentor import Augmentor
from core.augmentor_impl import Scale
from core.generator import SegmentsGenerator
from core.models import AVOLNet
from core.segments import Segment
from tests.utils import *


def map_dir(segments):
    return list(map(lambda s: s.dir, segments))


def map_raw(segments):
    return list(map(lambda s: '{}.mp4'.format(os.path.join(s.dir, s.ytid)), segments))


@pytest.fixture
def test_raw_file():
    return 'tests/data/segments/0qZ3tI4nAZE.mp4'


@pytest.fixture
def segment():
    with temp_dir('tests/.temp/segments/0qZ3tI4nAZE'):
        return Segment(root_dir='tests/.temp/segments',
                       ytid='0qZ3tI4nAZE',
                       start_seconds=6.000,
                       end_seconds=16.000,
                       positive_labels=['/m/07qrkrw', '/m/09x0r'])


@pytest.fixture
def segments(segment):
    return 100 * [segment]


@pytest.fixture
def model():
    return AVOLNet()


@pytest.fixture
def augmentor():
    return Augmentor(*(2 * [Scale(2)]))


def test_with_invalid_segments_type():
    with pytest.raises(TypeError):
        assert SegmentsGenerator(0, None)


def test_with_invalid_segments_item_type(model):
    with pytest.raises(TypeError):
        assert SegmentsGenerator([0], model)


def test_with_single_segment(test_raw_file, segment, model):
    with temp_dir(segment.dir):
        with temp_copy(test_raw_file, segment.dir):
            assert len(SegmentsGenerator(segment, model).segments) == 1


def test_with_empty_segments(model):
    with pytest.raises(ValueError):
        assert SegmentsGenerator([], model)


def test_with_unavailable_segments(segments, model):
    with pytest.raises(ValueError):
        assert SegmentsGenerator(segments, model)


def test_batch_size_default(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model)
            assert generator.batch_size == 16
            generator = SegmentsGenerator(segments, model, None)
            assert generator.batch_size == 16


def test_batch_size_invalid_type(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            with pytest.raises(TypeError):
                assert SegmentsGenerator(segments, model, '32')


def test_batch_size_invalid_value(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            with pytest.raises(ValueError):
                assert SegmentsGenerator(segments, model, -16)
            with pytest.raises(ValueError):
                assert SegmentsGenerator(segments, model, 0)


def test_vision_augmentor(test_raw_file, segments, model, augmentor):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, vision_augmentor=augmentor)
            assert generator.vision_augmentor is not None


def test_vision_augmentor_with_invalid_type(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            with pytest.raises(TypeError):
                assert SegmentsGenerator(segments, model, vision_augmentor=1)


def test_audio_augmentor(test_raw_file, segments, model, augmentor):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, audio_augmentor=augmentor)
            assert generator.audio_augmentor is not None


def test_audio_augmentor_with_invalid_type(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            with pytest.raises(TypeError):
                assert SegmentsGenerator(segments, model, audio_augmentor=1)


def test_default_augmentor(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model)
            assert generator.vision_augmentor is not None
            assert generator.audio_augmentor is not None


def test_augmentation_factor(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 16)
            assert generator.augmentation_factor == 1


def test_augmentation_factor_with_augmentor(test_raw_file, segments, model, augmentor):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 16, augmentor, augmentor)
            assert generator.augmentation_factor == len(augmentor) * len(augmentor)


def test_sample_size(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 16)
            assert generator.sample_size == 2 * generator.batch_size


def test_sample_size_with_augmentor(test_raw_file, segments, model, augmentor):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 16, augmentor, augmentor)
            assert generator.sample_size == 2 * generator.batch_size * generator.augmentation_factor


def test_augment_vision_should_match_model_vision_input_shape(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model)
            target_shape = model.vision_input_shape
            frame = np.ones((10, 10, 3))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_vision(frame)))
            frame = np.ones((10, 1000, 3))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_vision(frame)))
            frame = np.ones((1000, 10, 3))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_vision(frame)))
            frame = np.ones((1000, 1000, 3))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_vision(frame)))


def test_augment_vision_should_match_model_vision_input_shape_with_augmentor(test_raw_file, segments, model, augmentor):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, vision_augmentor=augmentor)
            target_shape = model.vision_input_shape
            frame = np.ones((10, 10, 3))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_vision(frame)))
            frame = np.ones((10, 1000, 3))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_vision(frame)))
            frame = np.ones((1000, 10, 3))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_vision(frame)))
            frame = np.ones((1000, 1000, 3))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_vision(frame)))


def test_augment_audio_should_match_model_audio_input_shape(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model)
            target_shape = model.audio_input_shape
            spectrogram = np.ones((10, 10, 1))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_audio(spectrogram)))
            spectrogram = np.ones((10, 1000, 1))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_audio(spectrogram)))
            spectrogram = np.ones((1000, 10, 1))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_audio(spectrogram)))
            spectrogram = np.ones((1000, 1000, 1))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_audio(spectrogram)))


def test_augment_audio_should_match_model_audio_input_shape_with_augmentor(test_raw_file, segments, model, augmentor):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, vision_augmentor=augmentor)
            target_shape = model.audio_input_shape
            spectrogram = np.ones((10, 10, 1))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_audio(spectrogram)))
            spectrogram = np.ones((10, 1000, 1))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_audio(spectrogram)))
            spectrogram = np.ones((1000, 10, 1))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_audio(spectrogram)))
            spectrogram = np.ones((1000, 1000, 1))
            assert all(map(lambda x: x.shape == target_shape, generator.augment_audio(spectrogram)))


def test_len(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 25)
            assert len(generator) == math.ceil(len(segments) / 25)


def test_len_should_ceil_the_batch_count(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 16)
            assert len(generator) == math.ceil(len(segments) / 16)


def test_model_with_invalid_type(test_raw_file, segments):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            with pytest.raises(TypeError):
                assert SegmentsGenerator(segments, 0)


def test_zip_samples(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 16)
            frames = np.ones((2 * generator.batch_size, *model.vision_input_shape))
            spectrograms = np.ones((2 * generator.batch_size, *model.audio_input_shape))
            labels = np.ones((2 * generator.batch_size, *model.output_shape, 1))
            samples = generator.zip_samples(frames, spectrograms, labels)
            assert len(samples) == generator.sample_size


def test_zip_samples_with_augmentor(test_raw_file, segments, model, augmentor):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 16, augmentor)
            frames = np.ones((2 * generator.batch_size * len(augmentor), *model.vision_input_shape))
            spectrograms = np.ones((2 * generator.batch_size * len(augmentor), *model.audio_input_shape))
            labels = np.ones((2 * generator.batch_size, *model.output_shape, 1))
            samples = generator.zip_samples(frames, spectrograms, labels)
            assert len(samples) == generator.sample_size


def test_getitem(test_raw_file, segments, model, augmentor):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 16, augmentor, augmentor)
            batch = generator[0]
            zipped = list(zip(*batch[0], *batch[1]))
            assert len(zipped) == generator.sample_size


def test_getitem_output_shape(test_raw_file, segments, model):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 16)
            batch_x, batch_y = generator[0]
            frames, spectrograms = batch_x
            labels, = batch_y
            assert all(map(lambda x: x.shape == model.vision_input_shape, frames))
            assert all(map(lambda x: x.shape == model.audio_input_shape, spectrograms))
            assert all(map(lambda x: x.shape == model.output_shape, labels))


def test_getitem_output_shape_with_augmentor(test_raw_file, segments, model, augmentor):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 16, augmentor, augmentor)
            batch_x, batch_y = generator[0]
            frames, spectrograms = batch_x
            labels, = batch_y
            assert all(map(lambda x: x.shape == model.vision_input_shape, frames))
            assert all(map(lambda x: x.shape == model.audio_input_shape, spectrograms))
            assert all(map(lambda x: x.shape == model.output_shape, labels))


def test_getitem_should_has_same_number_of_sample_for_all_inputs(test_raw_file, segments, model, augmentor):
    with temp_dir(segments[0].dir):
        with temp_copy(test_raw_file, segments[0].dir):
            generator = SegmentsGenerator(segments, model, 16, augmentor, augmentor)
            batch = generator[0]
            zipped = list(zip(*batch[0], *batch[1]))
            assert zipped is not None
