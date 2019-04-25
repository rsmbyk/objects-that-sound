import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from util import bit
from util.tensorplow import Ops


class Aug:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class Augmentor:
    def __init__(self, *augmentors):
        if not all(map(lambda aug: isinstance(aug, Aug), augmentors)):
            raise TypeError('\'augmentors\' must be of type {}'.format(Aug))

        if len(augmentors) == 0:
            raise ValueError('\'augmentors\' can not be empty')

        self.__augmentors = augmentors

    def __call__(self, item):
        augmented_items = list()

        for b in bit.bitstring(len(self.__augmentors)):
            x = item
            for i in range(len(self.__augmentors)):
                if b[i]:
                    x = self.__augmentors[i](x)
            augmented_items.append(np.array(x))

        return augmented_items

    def __len__(self):
        return pow(2, len(self.__augmentors))


class VisionAugmentor(Ops):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, frame):
        try:
            crop = tf.image.random_crop(frame, self.shape)
            brightness = tf.image.random_brightness(crop, 0.25)
            saturation = tf.image.random_saturation(brightness, 0.5, 1.5)
            flip = tf.image.random_flip_left_right(saturation)
            return flip
        except InvalidArgumentError:
            raise ValueError()


class AudioAugmentor(Ops):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, spectrogram):
        return tf.image.resize(spectrogram, self.shape)


class ValidationVisionAugmentor(VisionAugmentor):
    def __call__(self, frame):
        return tf.image.random_crop(frame, self.shape)
