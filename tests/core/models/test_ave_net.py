import pytest

from core.models import AVENet


@pytest.fixture
def ave_net():
    return AVENet()


def test_name(ave_net):
    assert ave_net.name == 'AVE-Net'


def test_vision_input_shape(ave_net):
    assert ave_net.vision_input_shape == (224, 224, 3)


def test_audio_input_shape(ave_net):
    assert ave_net.audio_input_shape == (257, 200, 1)


def test_output_shape(ave_net):
    assert ave_net.output_shape == (2,)
