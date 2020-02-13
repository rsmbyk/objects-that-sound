import os

import cv2
import numpy as np
import tensorflow as tf


class Ops:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


def load_image(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError('FILE ({})'.format(filename))

    image = cv2.imread(filename)
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    return image


def load_wav(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError('FILE ({})'.format(filename))

    contents = tf.io.read_file(filename)
    wav = tf.audio.decode_wav(contents)
    return wav


def save_image(image, outfile):
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))

    cast = tf.cast(image, tf.dtypes.uint8)
    encode = tf.image.encode_jpeg(cast)
    tf.io.write_file(outfile, encode)


def spectrogram(waveform, sample_rate, window_length, overlap):
    # calculate spectrogram properties
    window_size = int(sample_rate * window_length)
    stride = int(window_size * overlap)

    # compute the stft
    stft = tf.signal.stft(tf.squeeze(waveform), window_size, stride)

    # get the squared magnitude (spectrogram)
    spec = tf.square(tf.math.abs(stft))

    # convert to log-spectrogram
    log_spec = 10 * np.log10(spec + 1e-10)

    # Tensorflow spectrogram has time along y axis and frequencies along x axis
    # so we fix that
    log_spec_transposed = tf.transpose(log_spec)

    return log_spec_transposed
