import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from util.tensorplow import Ops


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
