from collections import Generator

import pytest

from util import bit


def test_bitstring_must_be_a_generator():
    assert isinstance(bit.bitstring(0), Generator)


def test_bitstring_len():
    assert len(list(bit.bitstring(4))) == 16


def test_bitstring_unique():
    assert len(set(list(map(str, bit.bitstring(4))))) == 16


def test_bitstring_with_negative_n():
    with pytest.raises(ValueError):
        assert bit.bitstring(-1)
