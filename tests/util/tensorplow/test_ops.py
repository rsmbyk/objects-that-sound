import pytest
import tensorflow as tf

from util.tensorplow import Ops


class Multiply(Ops):
    def __init__(self):
        super().__init__()
        self.x = tf.placeholder(tf.dtypes.float32)
        self.y = tf.placeholder(tf.dtypes.float32)

    def get_ops(self, *args, **kwargs):
        return tf.multiply(self.x, self.y)

    def parse(self, x, y):
        return {self.x: x, self.y: y}


class CustomKeyOps(Multiply):
    def get_key(self, x, y):
        return x % 2


class CustomArgsOps(Multiply):
    def clean_args(self, x, y, **kwargs):
        return [x, 1], kwargs


class NonDictParseOps(Ops):
    def __init__(self):
        super().__init__()
        self.x = tf.placeholder(tf.dtypes.float32)
        self.y = tf.placeholder(tf.dtypes.float32)

    def get_ops(self, *args, **kwargs):
        return tf.multiply(self.x, self.y)

    def parse(self, x, y):
        return 0


# noinspection PyAbstractClass
class UnimplementedOps(Ops):
    def parse(self, *args, **kwargs):
        return {}


# noinspection PyAbstractClass
class UnimplementedParse(Ops):
    def get_ops(self, *args, **kwargs):
        return tf.multiply(2, 3)


@pytest.fixture
def ops():
    return Multiply()


@pytest.fixture
def custom_key_ops():
    return CustomKeyOps()


@pytest.fixture
def custom_args_ops():
    return CustomArgsOps()


@pytest.fixture
def non_dict_parse_ops():
    return NonDictParseOps()


@pytest.fixture
def unimplemented_ops():
    return UnimplementedOps()


@pytest.fixture
def unimplemented_parse():
    return UnimplementedParse()


def test_parse_must_return_dict(non_dict_parse_ops):
    with pytest.raises(TypeError):
        assert non_dict_parse_ops(2, 3)


def test_ops_must_be_implemented(unimplemented_ops):
    with pytest.raises(NotImplementedError):
        assert unimplemented_ops()


def test_parse_must_be_implemented(unimplemented_parse):
    with pytest.raises(NotImplementedError):
        assert unimplemented_parse()


def test_get_ops_should_return_same_reference(ops):
    assert ops.ops() is ops.ops()


def test_with_undefined_custom_key(ops):
    assert ops.ops(1) is ops.ops(2)


def test_custom_key_ops(custom_key_ops):
    assert custom_key_ops.ops(2, 2) is not custom_key_ops.ops(3, 2)


def test_should_return_same_object_for_same_key(custom_key_ops):
    assert custom_key_ops.ops(2, 2) is custom_key_ops.ops(2, 4)


def test_custom_args(custom_args_ops):
    assert custom_args_ops(3, 4) == 3


def test_custom_kwargs(custom_args_ops):
    assert custom_args_ops(x=5, y=4) == 5


def test_call(ops):
    assert ops(2, 3) == 6
