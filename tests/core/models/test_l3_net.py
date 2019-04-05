import pytest

from core.models import L3Net


@pytest.fixture
def l3_net():
    return L3Net()


def test_vision_input_shape(l3_net):
    assert l3_net.vision_input_shape == (224, 224, 3)


def test_audio_input_shape(l3_net):
    assert l3_net.audio_input_shape == (257, 199, 1)


def test_output_shape(l3_net):
    assert l3_net.output_shape == (2,)
