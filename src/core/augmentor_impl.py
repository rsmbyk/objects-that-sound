import random

import tensorflow as tf

from core.augmentor import Aug


class Scale(Aug):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor


class RandomScale(Aug):
    def __init__(self, factor, center=True):
        self.a = -factor if center else 0
        self.b = factor

    def __call__(self, x):
        return x * random.uniform(self.a, self.b)


class Resize(Aug):
    def __init__(self, *size):
        self.size = size

    def __call__(self, image):
        return tf.image.resize(image, self.size)


class RandomCrop(Aug):
    def __init__(self, *size):
        self.size = size

    def __call__(self, image):
        return tf.image.random_crop(image, self.size)


class RandomSaturation(Aug):
    def __init__(self, max_delta):
        self.max_delta = max_delta

    def __call__(self, image):
        return tf.image.random_saturation(image, 0, self.max_delta)


class RandomBrightness(Aug):
    def __init__(self, max_delta):
        self.max_delta = max_delta

    def __call__(self, image):
        return tf.image.random_brightness(image, self.max_delta)


class RandomHorizontalFlip(Aug):
    def __call__(self, image):
        return tf.image.random_flip_left_right(image)
