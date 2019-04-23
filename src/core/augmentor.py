import tensorflow as tf

from util.tensorplow import Ops


class VisionAugmentor(Ops):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    # noinspection PyAttributeOutsideInit
    def set_placeholder(self):
        self.frame = tf.placeholder(tf.dtypes.uint8, (None, None, 3))

    def get_ops(self, frame):
        crop = tf.image.random_crop(self.frame, self.shape)
        brightness = tf.image.random_brightness(crop, 0.25)
        saturation = tf.image.random_saturation(brightness, 0.5, 1.5)
        flip = tf.image.random_flip_left_right(saturation)
        return flip

    def parse(self, frame):
        return {self.frame: frame}


class AudioAugmentor(Ops):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    # noinspection PyAttributeOutsideInit
    def set_placeholder(self):
        self.spectrogram = tf.placeholder(tf.dtypes.uint8, (None, None, 1))

    def get_ops(self, spectrogram):
        resize = tf.image.resize_images(self.spectrogram, self.shape)
        return resize

    def parse(self, spectrogram):
        return {self.spectrogram: spectrogram}


class ValidationVisionAugmentor(VisionAugmentor):
    def get_ops(self, frame):
        return tf.image.random_crop(self.frame, self.shape)
