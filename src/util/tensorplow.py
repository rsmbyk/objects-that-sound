import os

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops


def run(tensor, feed_dict=None):
    with tf.Session() as sess:
        return sess.run(tensor,  feed_dict)


class Ops:
    def __init__(self):
        self.__ops = dict()

    def get_ops(self, *args, **kwargs):
        raise NotImplementedError()

    def get_key(self, *args, **kwargs):
        return None

    def parse(self, *args, **kwargs):
        raise NotImplementedError()

    def clean_args(self, *args, **kwargs):
        return args, kwargs

    def ops(self, *args, **kwargs):
        ops_key = self.get_key(*args, **kwargs)
        if ops_key not in self.__ops:
            self.__ops[ops_key] = self.get_ops(*args, **kwargs)
        return self.__ops[ops_key]

    def __call__(self, *args, **kwargs):
        args, kwargs = self.clean_args(*args, **kwargs)
        ops = self.ops(*args, **kwargs)
        feed_dict = self.parse(*args, **kwargs)
        if not isinstance(feed_dict, dict):
            raise TypeError('\'parse\' result must be a dict')
        return run(ops, feed_dict)


def load_image(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError('FILE ({})'.format(filename))

    contents = tf.read_file(filename)
    image = tf.image.decode_image(contents)
    return image


def load_wav(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError('FILE ({})'.format(filename))

    audio_binary = tf.read_file(filename)
    wav = audio_ops.decode_wav(audio_binary)
    return wav


def save_image(image, outfile):
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))

    with open(outfile, 'wb') as output:
        cast = tf.cast(image, tf.dtypes.uint8)
        encode = tf.image.encode_png(cast)
        output.write(run(encode))


def spectrogram(waveform, sample_rate, window_length, overlap):
    # calculate spectrogram properties
    window_size = sample_rate * window_length
    stride = window_size * overlap

    # compute the spectrogram
    spc = audio_ops.audio_spectrogram(waveform, window_size, stride)

    # custom brightness
    mul = tf.multiply(spc, 100)

    # normalize pixels
    minimum = tf.minimum(mul, 255)

    # expand dims so we get the proper shape
    expand_dims = tf.expand_dims(minimum, -1)

    # Tensorflow spectrogram has time along y axis and frequencies along x axis
    # so we fix that
    flip = tf.image.flip_left_right(expand_dims)
    transpose = tf.image.transpose_image(flip)

    # remove the trailing dimension
    squeeze = tf.squeeze(transpose, 0)
    return squeeze
