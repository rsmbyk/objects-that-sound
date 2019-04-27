import os

import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops


class Ops:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


def load_image(filename, channels=None):
    if not os.path.exists(filename):
        raise FileNotFoundError('FILE ({})'.format(filename))

    contents = tf.io.read_file(filename)
    image = tf.image.decode_image(contents, channels)
    return image


def load_wav(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError('FILE ({})'.format(filename))

    contents = tf.io.read_file(filename)
    wav = gen_audio_ops.decode_wav(contents)
    return wav


def save_image(image, outfile):
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))

    cast = tf.cast(image, tf.dtypes.uint8)
    encode = tf.image.encode_jpeg(cast)
    tf.io.write_file(outfile, encode)


def spectrogram(waveform, sample_rate, window_length, overlap):
    # calculate spectrogram properties
    window_size = sample_rate * window_length
    stride = window_size * overlap

    # compute the spectrogram
    spc = gen_audio_ops.audio_spectrogram(waveform, window_size, stride)

    # expand dims so we get the proper shape
    expand_dims = tf.expand_dims(spc, -1)

    # Tensorflow spectrogram has time along y axis and frequencies along x axis
    # so we fix that
    flip = tf.image.flip_left_right(expand_dims)
    transpose = tf.image.transpose(flip)

    # remove the trailing dimension
    squeeze = tf.squeeze(transpose, 0)
    return squeeze
