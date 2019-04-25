import pytest
import tensorflow as tf

from core.augmentor import Aug


class TestAug(Aug):
    def __call__(self, a, b):
        return tf.multiply(a, b).numpy()


# noinspection PyAbstractClass
class UnimplementedAug(Aug):
    pass


@pytest.fixture
def aug():
    return TestAug()


@pytest.fixture
def unimplemented_aug():
    return UnimplementedAug()


def test_aug_should_not_be_implemented():
    with pytest.raises(NotImplementedError):
        assert Aug()()


def test_call(aug):
    assert aug(2, 3) == 6


def test_aug_implementation_must_implement_call_method(unimplemented_aug):
    with pytest.raises(NotImplementedError):
        assert unimplemented_aug()
