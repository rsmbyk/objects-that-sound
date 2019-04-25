import pytest
import tensorflow as tf

from util.tensorplow import Ops


class TestOps(Ops):
    def __call__(self, x, y):
        return tf.multiply(x, y)


# noinspection PyAbstractClass
class UnimplementedOps(Ops):
    pass


@pytest.fixture
def ops():
    return TestOps()


@pytest.fixture
def unimplemented_ops():
    return UnimplementedOps()


def test_ops_must_be_implemented(unimplemented_ops):
    with pytest.raises(NotImplementedError):
        assert unimplemented_ops()


def test_call(ops):
    assert ops(2, 3).numpy() == 6
