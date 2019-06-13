import random
from functools import reduce

import cv2
import math
import numpy as np
from tensorflow.python.keras.utils import Sequence

from core.augmentor import Augmentor
from core.models import AVC
from core.segments import SegmentsWrapper, Segment


class SegmentsGenerator(Sequence):
    default_augmentor = Augmentor()

    def __init__(self, segments, model, batch_size=None, vision_augmentor=None, audio_augmentor=None):
        self.__validate_segments(segments)
        self.__validate_batch_size(batch_size)
        self.__validate_model(model)
        self.__validate_augmentors(vision_augmentor, audio_augmentor)

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

    def __validate_augmentors(self, vision_augmentor, audio_augmentor):
        if vision_augmentor is not None and not isinstance(vision_augmentor, Augmentor):
            raise TypeError('\'vision_augmentor\' must be of type {}, not {}'.format(Augmentor, type(vision_augmentor)))

        if audio_augmentor is not None and not isinstance(audio_augmentor, Augmentor):
            raise TypeError('\'vision_augmentor\' must be of type {}, not {}'.format(Augmentor, type(audio_augmentor)))

        self.vision_augmentor = vision_augmentor or self.default_augmentor
        self.audio_augmentor = audio_augmentor or self.default_augmentor
        self.augmentation_factor = len(self.vision_augmentor) * len(self.audio_augmentor)
        self.sample_size = 2 * self.batch_size * self.augmentation_factor

    def augment_vision(self, frame):
        if min(np.subtract(frame.shape, self.model.vision_input_shape)[:-1]) >= 0:
            crop_space = np.subtract(frame.shape, self.model.vision_input_shape)[:-1]
            crop_start = list(map(lambda x: random.randint(0, x), crop_space))
            x1, y1 = crop_start
            x2, y2 = list(map(lambda x: x + self.model.vision_input_shape[0], crop_start))
            out = frame[x1:x2, y1:y2]
        else:
            out = cv2.resize(frame, self.model.vision_input_shape[:-1])

        return self.vision_augmentor(out)

    def augment_audio(self, spectrogram):
        out = cv2.resize(spectrogram, tuple(reversed(self.model.audio_input_shape[:-1])))
        out = np.expand_dims(out, -1)
        return self.audio_augmentor(out)

    def zip_samples(self, frames, spectrograms, labels):
        samples = list()
        for i in range(len(labels)):
            for v in range(len(self.vision_augmentor)):
                for a in range(len(self.audio_augmentor)):
                    samples.append((frames[v + i * len(self.vision_augmentor)],
                                    spectrograms[a + i * len(self.audio_augmentor)],
                                    labels[i]))
        return samples

    def __getitem__(self, index):
        batch_slice = slice(index * self.batch_size, (index+1) * self.batch_size)
        batch_segments = self.segments[batch_slice]
        batch = list()

        for i, segment in enumerate(batch_segments):
            batch.append((segment.get_positive_sample(), 1))

            positive_frame = segment.get_positive_sample()[0]
            other_segments = list(range(0, i)) + list(range(i+1, len(self.segments)))
            negative_pair = self.segments[random.choice(other_segments)]
            negative_audio = negative_pair.get_positive_sample()[1]
            batch.append(((positive_frame, negative_audio), 0))

        random.shuffle(batch)
        batch_x, batch_y = zip(*batch)
        frames, spectrograms = zip(*batch_x)

        frames = list(reduce(list.__add__, list(map(self.augment_vision, frames))))
        spectrograms = list(reduce(list.__add__, list(map(self.augment_audio, spectrograms))))
        labels = list(map(lambda y: np.reshape(y, self.model.output_shape), batch_y))

        samples = self.zip_samples(frames, spectrograms, labels)
        frames, spectrograms, labels = zip(*samples)

        return [frames, spectrograms], [labels]

    def on_epoch_end(self):
        random.shuffle(self.segments)

    def __len__(self):
        return math.ceil(len(self.segments) / self.batch_size)
