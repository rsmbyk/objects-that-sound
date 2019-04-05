import pytest

from core.models import AVOLNet


@pytest.fixture
def avol_net():
    return AVOLNet()


def test_vision_input_shape(avol_net):
    assert avol_net.vision_input_shape == (224, 224, 3)


def test_audio_input_shape(avol_net):
    assert avol_net.audio_input_shape == (257, 200, 1)


def test_output_shape(avol_net):
    assert avol_net.output_shape == (1,)
