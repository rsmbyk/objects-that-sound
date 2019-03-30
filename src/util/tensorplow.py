import os

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops


def run(tensor):
    with tf.Session() as sess:
        return sess.run(tensor)


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
