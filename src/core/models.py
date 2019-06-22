from functools import reduce

import tensorflow as tf
from tensorflow.python.keras import Model, Sequential, backend as K
from tensorflow.python.keras.activations import sigmoid
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.metrics import binary_accuracy


class AVC:
    def __init__(self):
        self.__model__ = None
        self.__vision_subnetwork = None
        self.__audio_subnetwork = None
        self.__fusion_subnetwork = None

    def get_model(self) -> Model:
        if self.__model__ is None:
            self.__model__ = self.__fuse__()
        return self.__model__

    def compile(self, lr=0.00001, decay=0.00001):
        optimizer = tf.optimizers.Adam(lr=lr, decay=decay)
        self.get_model().compile(optimizer,
                                 loss=binary_crossentropy,
                                 metrics=[binary_accuracy])
        return self.get_model()

    @property
    def name(self):
        raise NotImplementedError('model\'s name must be defined')

    @property
    def vision_subnetwork(self):
        if self.__vision_subnetwork is None:
            self.__vision_subnetwork = self.get_vision_subnetwork()
        return self.__vision_subnetwork

    @property
    def audio_subnetwork(self):
        if self.__audio_subnetwork is None:
            self.__audio_subnetwork = self.get_audio_subnetwork()
        return self.__audio_subnetwork

    @property
    def fusion_subnetwork(self):
        if self.__fusion_subnetwork is None:
            self.__fusion_subnetwork = self.get_fusion_subnetwork()
        return self.__fusion_subnetwork

    def get_vision_subnetwork(self):
        raise NotImplementedError('vision subnetwork must be defined')

    def get_audio_subnetwork(self):
        raise NotImplementedError('audio subnetwork must be defined')

    def get_fusion_subnetwork(self):
        raise NotImplementedError('fusion subnetwork must be defined')

    @property
    def vision_input_shape(self):
        return self.vision_subnetwork.input_shape[1:]

    @property
    def audio_input_shape(self):
        return self.audio_subnetwork.input_shape[1:]

    @property
    def input_shape(self):
        return [self.vision_input_shape, self.audio_input_shape]

    @property
    def output_shape(self):
        return self.get_model().output_shape[1:]

    def __fuse__(self):
        concat = [self.vision_subnetwork.output, self.audio_subnetwork.output]
        fusion = reduce(lambda x, f: f(x), self.fusion_subnetwork, concat)
        inputs = [self.vision_subnetwork.input, self.audio_subnetwork.input]
        return Model(inputs, fusion, name=self.name)


class L3Net(AVC):
    @property
    def name(self):
        return 'L3-Net'

    def get_vision_subnetwork(self):
        return Sequential([
            InputLayer((224, 224, 3), name='vision_input'),

            Conv2D(64, 3, padding='same', name='vision_conv1_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(64, 3, padding='same', name='vision_conv1_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(2, name='vision_pool1'),

            Conv2D(128, 3, padding='same', name='vision_conv2_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(128, 3, padding='same', name='vision_conv2_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(2, name='vision_pool2'),

            Conv2D(256, 3, padding='same', name='vision_conv3_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(256, 3, padding='same', name='vision_conv3_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(2, name='vision_pool3'),

            Conv2D(512, 3, padding='same', name='vision_conv4_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(512, 3, padding='same', name='vision_conv4_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(28, name='vision_pool4'),
            Flatten(name='vision_flatten')])

    def get_audio_subnetwork(self):
        return Sequential([
            InputLayer((257, 199, 1), name='audio_input'),

            Conv2D(64, 3, padding='same', name='audio_conv1_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(64, 3, padding='same', name='audio_conv1_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(2, name='audio_pool1'),

            Conv2D(128, 3, padding='same', name='audio_conv2_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(128, 3, padding='same', name='audio_conv2_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(2, name='audio_pool2'),

            Conv2D(256, 3, padding='same', name='audio_conv3_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(256, 3, padding='same', name='audio_conv3_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(2, name='audio_pool3'),

            Conv2D(512, 3, padding='same', name='audio_conv4_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(512, 3, padding='same', name='audio_conv4_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((32, 24), name='audio_pool4'),
            Flatten(name='audio_flatten')])

    def get_fusion_subnetwork(self):
        return [
            Concatenate(name='concat'),
            Dense(128, name='fc1'),
            Dense(2, name='fc2'),
            Softmax(name='softmax')]


class AVENet(AVC):
    @property
    def name(self):
        return 'AVE-Net'

    def get_vision_subnetwork(self):
        return Sequential([
            InputLayer((224, 224, 3), name='vision_input'),

            Conv2D(64, 3, 2, padding='same', name='vision_conv1_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(64, 3, padding='same', name='vision_conv1_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(2, name='vision_pool1'),

            Conv2D(128, 3, padding='same', name='vision_conv2_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(128, 3, padding='same', name='vision_conv2_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(2, name='vision_pool2'),

            Conv2D(256, 3, padding='same', name='vision_conv3_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(256, 3, padding='same', name='vision_conv3_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(2, name='vision_pool3'),

            Conv2D(512, 3, padding='same', name='vision_conv4_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(512, 3, padding='same', name='vision_conv4_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(14, name='vision_pool4'),

            Dense(128, name='vision_fc1'),
            ReLU(),
            Dense(128, name='vision_fc2'),
            Lambda(lambda x: K.l2_normalize(x), name='vision_L2_norm')])

    def get_audio_subnetwork(self):
        return Sequential([
            InputLayer((257, 200, 1), name='audio_input'),

            Conv2D(64, 3, 2, padding='same', name='audio_conv1_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(64, 3, padding='same', name='audio_conv1_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(2, name='audio_pool1'),

            Conv2D(128, 3, padding='same', name='audio_conv2_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(128, 3, padding='same', name='audio_conv2_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(2, name='audio_pool2'),

            Conv2D(256, 3, padding='same', name='audio_conv3_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(256, 3, padding='same', name='audio_conv3_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(2, name='audio_pool3'),

            Conv2D(512, 3, padding='same', name='audio_conv4_1'),
            BatchNormalization(),
            ReLU(),
            Conv2D(512, 3, padding='same', name='audio_conv4_2'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((16, 12), name='audio_pool4'),

            Dense(128, name='audio_fc1'),
            ReLU(),
            Dense(128, name='audio_fc2'),
            Lambda(lambda x: K.l2_normalize(x), name='audio_L2_norm')])

    def get_fusion_subnetwork(self):
        return [
            Lambda(lambda x: K.sqrt(K.sum(K.square(x[0] - x[1]), axis=-1)), name='euclidean_distance'),
            Dense(2, name='fc3'),
            Flatten(name='vision_flatten'),
            Softmax(name='softmax')]


class AVOLNet(AVC):
    @property
    def name(self):
        return 'AVOL-Net'

    def get_vision_subnetwork(self):
        return Sequential([
            InputLayer((224, 224, 3), name='vision_input'),

            ZeroPadding2D(name='vision_zero_padding1_1'),
            Conv2D(filters=64, kernel_size=3, strides=2, name='vision_conv1_1'),
            BatchNormalization(name='vision_batch_norm1_1'),
            ReLU(name='vision_relu1_1'),
            ZeroPadding2D(name='vision_zero_padding1_2'),
            Conv2D(filters=64, kernel_size=3, name='vision_conv1_2'),
            BatchNormalization(name='vision_batch_norm1_2'),
            ReLU(name='vision_relu1_2'),
            MaxPool2D(2, name='vision_pool1'),

            ZeroPadding2D(name='vision_zero_padding2_1'),
            Conv2D(filters=128, kernel_size=3, name='vision_conv2_1'),
            BatchNormalization(name='vision_batch_norm2_1'),
            ReLU(name='vision_relu2_1'),
            ZeroPadding2D(name='vision_zero_padding2_2'),
            Conv2D(filters=128, kernel_size=3, name='vision_conv2_2'),
            BatchNormalization(name='vision_batch_norm2_2'),
            ReLU(name='vision_relu2_2'),
            MaxPool2D(2, name='vision_pool2'),

            ZeroPadding2D(name='vision_zero_padding3_1'),
            Conv2D(filters=256, kernel_size=3, name='vision_conv3_1'),
            BatchNormalization(name='vision_batch_norm3_1'),
            ReLU(name='vision_relu3_1'),
            ZeroPadding2D(name='vision_zero_padding3_2'),
            Conv2D(filters=256, kernel_size=3, name='vision_conv3_2'),
            BatchNormalization(name='vision_batch_norm3_2'),
            ReLU(name='vision_relu3_2'),
            MaxPool2D(2, name='vision_pool3'),

            ZeroPadding2D(name='vision_zero_padding4_1'),
            Conv2D(filters=512, kernel_size=3, name='vision_conv4_1'),
            BatchNormalization(name='vision_batch_norm4_1'),
            ReLU(name='vision_relu4_1'),
            ZeroPadding2D(name='vision_zero_padding4_2'),
            Conv2D(filters=512, kernel_size=3, name='vision_conv4_2'),
            BatchNormalization(name='vision_batch_norm4_2'),
            ReLU(name='vision_relu4_2'),

            Conv2D(filters=128, kernel_size=1, name='vision_conv5'),
            BatchNormalization(name='vision_batch_norm5'),
            ReLU(name='vision_relu5'),
            Conv2D(filters=128, kernel_size=1, name='vision_conv6')])

    def get_audio_subnetwork(self):
        return Sequential([
            InputLayer((257, 200, 1), name='audio_input'),

            ZeroPadding2D(name='audio_zero_padding1_1'),
            Conv2D(filters=64, kernel_size=3, strides=2, name='audio_conv1_1'),
            BatchNormalization(name='audio_batch_norm1_1'),
            ReLU(name='audio_relu1_1'),
            ZeroPadding2D(name='audio_zero_padding1_2'),
            Conv2D(filters=64, kernel_size=3, name='audio_conv1_2'),
            BatchNormalization(name='audio_batch_norm1_2'),
            ReLU(name='audio_relu1_2'),
            MaxPool2D(2, name='audio_pool1'),

            ZeroPadding2D(name='audio_zero_padding2_1'),
            Conv2D(filters=128, kernel_size=3, name='audio_conv2_1'),
            BatchNormalization(name='audio_batch_norm2_1'),
            ReLU(name='audio_relu2_1'),
            ZeroPadding2D(name='audio_zero_padding2_2'),
            Conv2D(filters=128, kernel_size=3, name='audio_conv2_2'),
            BatchNormalization(name='audio_batch_norm2_2'),
            ReLU(name='audio_relu2_2'),
            MaxPool2D(2, name='audio_pool2'),

            ZeroPadding2D(name='audio_zero_padding3_1'),
            Conv2D(filters=256, kernel_size=3, name='audio_conv3_1'),
            BatchNormalization(name='audio_batch_norm3_1'),
            ReLU(name='audio_relu3_1'),
            ZeroPadding2D(name='audio_zero_padding3_2'),
            Conv2D(filters=256, kernel_size=3, name='audio_conv3_2'),
            BatchNormalization(name='audio_batch_norm3_2'),
            ReLU(name='audio_relu3_2'),
            MaxPool2D(2, name='audio_pool3'),

            ZeroPadding2D(name='audio_zero_padding4_1'),
            Conv2D(filters=512, kernel_size=3, name='audio_conv4_1'),
            BatchNormalization(name='audio_batch_norm4_1'),
            ReLU(name='audio_relu4_1'),
            ZeroPadding2D(name='audio_zero_padding4_2'),
            Conv2D(filters=512, kernel_size=3, name='audio_conv4_2'),
            BatchNormalization(name='audio_batch_norm4_2'),
            ReLU(name='audio_relu4_2'),
            MaxPool2D((16, 12), name='audio_pool4'),

            Dense(128, name='audio_fc1'),
            ReLU(name='audio_relu5'),
            Dense(128, name='audio_fc2')])

    def get_fusion_subnetwork(self):
        return [
            Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=True), name='scalar_products'),
            Conv2D(1, 1, padding='same', name='conv7'),
            Activation(sigmoid, name='sigmoid'),
            MaxPool2D(14, name='maxpool'),
            Flatten()]


def retrieve_model(name):
    if name == 'l3':
        return L3Net
    if name == 'ave':
        return AVENet
    if name == 'avol':
        return AVOLNet
    raise ValueError('Unknown model \'{}\''.format(name))
