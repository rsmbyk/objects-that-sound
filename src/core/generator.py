import random

import math
import numpy as np
from tensorflow.python.keras.utils import Sequence

import core.augmentor as aug
from core.models import AVC
from core.segments import SegmentsWrapper, Segment


class SegmentsGenerator(Sequence):
    def __init__(self, segments, model, batch_size=None):
        self.__validate_segments(segments)
        self.__validate_batch_size(batch_size)
        self.__validate_model(model)
        self.__initialize_augmentors()

    def __validate_segments(self, segments):
        allowed_segments_types = (SegmentsWrapper, list, Segment)
        if not isinstance(segments, allowed_segments_types):
            raise TypeError('\'segments\' must be of type {} or {}'.format(*allowed_segments_types))

        if isinstance(segments, Segment):
            segments = [segments]

        if not all(map(lambda x: isinstance(x, Segment), segments)):
            raise TypeError('all items in \'segments\' must be of type {}'.format(Segment))

        if len(segments) == 0:
            raise ValueError('\'segments\' can\'t be empty')

        segments = list(filter(lambda x: x.is_available, segments))

        if len(segments) == 0:
            raise ValueError('all the segment in \'segments\' are not available')

        self.segments = segments

    def __validate_batch_size(self, batch_size):
        if batch_size is None:
            batch_size = 16

        if not isinstance(batch_size, int):
            raise TypeError('\'batch_size\' must be an integer')

        if batch_size <= 0:
            raise ValueError('\'batch_size\' must be a positive value')

        self.batch_size = batch_size

    def __validate_model(self, model: AVC):
        if not isinstance(model, AVC):
            raise TypeError('\'model\' must be of type {}, not {}'.format(AVC, type(model)))

        self.model = model

    def __initialize_augmentors(self):
        self.frame_augmentor = aug.VisionAugmentor(self.model.vision_input_shape)
        self.audio_augmentor = aug.AudioAugmentor(self.model.audio_input_shape[:2])

    def __getitem__(self, index):
        batch_slice = slice(index * self.batch_size, (index+1) * self.batch_size)
        batch_segments = self.segments[batch_slice]
        batch = list()

        for segment in batch_segments:
            batch.append((segment.get_positive_sample(), 1))

            try:
                batch.append((segment.get_negative_sample(), 0))
            except IndexError:
                """
                This exception happens if the segment does not have
                negative sample (start_seconds and end_seconds
                cover the whole video)
                """

        random.shuffle(batch)
        batch_x, batch_y = zip(*batch)
        frames, spectrograms = zip(*batch_x)

        frames = np.array(list(map(self.frame_augmentor, frames)))
        spectrograms = np.array(list(map(self.audio_augmentor, spectrograms)))
        labels = np.array(batch_y)

        return [frames, spectrograms], labels

    def __len__(self):
        return math.ceil(len(self.segments) / self.batch_size)


class ValidationGenerator(SegmentsGenerator):
    def __initialize_augmentors(self):
        self.frame_augmentor = aug.ValidationVisionAugmentor(self.model.vision_input_shape)
        self.audio_augmentor = aug.AudioAugmentor(self.model.audio_input_shape[:2])
