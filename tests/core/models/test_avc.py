import numpy as np
import pytest
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import *

from core.models import AVC


@pytest.fixture
def avc():
    return AVC()


@pytest.fixture
def model():
    class CompleteAVC(AVC):
        @property
        def name(self):
            return 'test_model'

        def get_vision_subnetwork(self):
            return Sequential([InputLayer((224, 224, 3)), MaxPool2D(224)])

        def get_audio_subnetwork(self):
            return Sequential([InputLayer((257, 199, 1)), MaxPool2D((257, 199))])

        def get_fusion_subnetwork(self):
            return [Concatenate(), Dense(1), Flatten()]

    return CompleteAVC()


# noinspection PyAbstractClass
@pytest.fixture
def incomplete_model():
    class IncompleteAVC(AVC):
        pass

    return IncompleteAVC()


def test_name_should_not_be_implemented(avc):
    with pytest.raises(NotImplementedError):
        assert avc.name


def test_vision_subnetwork_should_not_be_implemented(avc):
    with pytest.raises(NotImplementedError):
        assert avc.vision_subnetwork


def test_audio_subnetwork_should_not_be_implemented(avc):
    with pytest.raises(NotImplementedError):
        assert avc.audio_subnetwork


def test_fusion_subnetwork_should_not_be_implemented(avc):
    with pytest.raises(NotImplementedError):
        assert avc.fusion_subnetwork


def test_avc_model_should_defines_name(incomplete_model):
    with pytest.raises(NotImplementedError):
        assert incomplete_model.name


def test_avc_model_should_implements_all_subnetwork(incomplete_model):
    with pytest.raises(NotImplementedError):
        assert incomplete_model.vision_subnetwork

    with pytest.raises(NotImplementedError):
        assert incomplete_model.audio_subnetwork

    with pytest.raises(NotImplementedError):
        assert incomplete_model.fusion_subnetwork


def test_name(model):
    assert model.name == 'test_model'


def test_vision_subnetwork_should_return_same_object(model):
    assert model.vision_subnetwork is model.vision_subnetwork


def test_audio_subnetwork_should_return_same_object(model):
    assert model.audio_subnetwork is model.audio_subnetwork


def test_fusion_subnetwork_should_return_same_object(model):
    assert model.fusion_subnetwork is model.fusion_subnetwork


def test_model_should_return_same_object(model):
    assert model.get_model() is model.get_model()


def test_vision_input_shape(model):
    assert model.vision_input_shape == (224, 224, 3)


def test_audio_input_shape(model):
    assert model.audio_input_shape == (257, 199, 1)


def test_output_shape(model):
    assert model.output_shape == (1,)


def test_input_shape(model):
    input_shape = [model.vision_input_shape, model.audio_input_shape]
    assert model.input_shape == input_shape


def test_compile(model):
    model = model.compile()
    assert model.optimizer is not None
    assert model.loss is not None


def test_compile_with_custom_lr_and_decay(model):
    model = model.compile(0.1, 0.01)
    assert np.isclose(model.optimizer.lr.numpy(), 0.1)
    assert np.isclose(model.optimizer.decay.numpy(), 0.01)
