import pytest

from core import models


def test_retrieve_model_l3():
    model = models.retrieve_model('l3')
    assert type(model()) == models.L3Net


def test_retrieve_model_ave():
    model = models.retrieve_model('ave')
    assert type(model()) == models.AVENet


def test_retrieve_model_avol():
    model = models.retrieve_model('avol')
    assert type(model()) == models.AVOLNet


def test_retrieve_model_unknown():
    with pytest.raises(ValueError):
        models.retrieve_model('unknown')
