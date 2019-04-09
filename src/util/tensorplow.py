import os

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops

tp_ops = dict()
tp_graph = tf.Graph()


def run(tensor, feed_dict=None):
    with tf.Session() as sess:
        return sess.run(tensor, feed_dict)


class Ops:
    def __init__(self):
        self.__ops = dict()
        with tp_graph.as_default():
            self.set_placeholder()

    def set_placeholder(self):
        pass

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
            with tp_graph.as_default():
                self.__ops[ops_key] = self.get_ops(*args, **kwargs)
        return self.__ops[ops_key]

    def __call__(self, *args, **kwargs):
        args, kwargs = self.clean_args(*args, **kwargs)
        ops = self.ops(*args, **kwargs)
        feed_dict = self.parse(*args, **kwargs)
        if not isinstance(feed_dict, dict):
            raise TypeError('\'parse\' result must be a dict')
        with tp_graph.as_default():
            return run(ops, feed_dict)


def get_ops(class_):
    def outer_decorator(func):
        def inner_decorator(*args, **kwargs):
            if class_ not in tp_ops:
                tp_ops[class_] = class_()
            return func(tp_ops[class_], *args, **kwargs)
        return inner_decorator
    return outer_decorator


class LoadImage(Ops):
    # noinspection PyAttributeOutsideInit
    def set_placeholder(self):
        self.filename = tf.placeholder(tf.dtypes.string)

    def get_ops(self, filename, channels):
        contents = tf.read_file(self.filename)
        image = tf.image.decode_image(contents, channels)
        return image

    def get_key(self, filename, channels):
        return channels

    def clean_args(self, filename, channels=None, **kwargs):
        return [filename, channels], kwargs

    def parse(self, filename, channels):
        return {self.filename: filename}


class LoadWav(Ops):
    # noinspection PyAttributeOutsideInit
    def set_placeholder(self):
        self.filename = tf.placeholder(tf.dtypes.string)

    def get_ops(self, filename):
        contents = tf.read_file(self.filename)
        wav = audio_ops.decode_wav(contents)
        return wav

    def parse(self, filename):
        return {self.filename: filename}


class EncodePNG(Ops):
    # noinspection PyAttributeOutsideInit
    def set_placeholder(self):
        self.image = tf.placeholder(tf.dtypes.uint8, (None, None, None))

    def get_ops(self, image):
        cast = tf.cast(self.image, tf.dtypes.uint8)
        encode = tf.image.encode_png(cast)
        return encode

    def parse(self, image):
        return {self.image: image}


class Spectrogram(Ops):
    # noinspection PyAttributeOutsideInit
    def set_placeholder(self):
        self.waveform = tf.placeholder(tf.dtypes.float32)

    def get_ops(self, waveform, window_size, stride):
        # compute the spectrogram
        spc = audio_ops.audio_spectrogram(self.waveform, window_size, stride)

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

    def get_key(self, waveform, window_size, stride):
        return window_size, stride

    def parse(self, waveform, window_size, stride):
        return {self.waveform: waveform}


@get_ops(LoadImage)
def load_image(ops, filename, channels=None):
    if not os.path.exists(filename):
        raise FileNotFoundError('FILE ({})'.format(filename))

    return ops(filename, channels)


@get_ops(LoadWav)
def load_wav(ops, filename):
    if not os.path.exists(filename):
        raise FileNotFoundError('FILE ({})'.format(filename))

    return ops(filename)


@get_ops(EncodePNG)
def save_image(ops, image, outfile):
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))

    with open(outfile, 'wb') as output:
        output.write(ops(image))


@get_ops(Spectrogram)
def spectrogram(ops, waveform, sample_rate, window_length, overlap):
    # calculate spectrogram properties
    window_size = sample_rate * window_length
    stride = window_size * overlap

    return ops(waveform, window_size, stride)
