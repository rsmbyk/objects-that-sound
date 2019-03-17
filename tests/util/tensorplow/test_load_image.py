import pytest

from util import tensorplow as tp


@pytest.fixture
def test_image_file():
    return 'tests/data/tensorplow/test.jpg'


@pytest.fixture
def non_existing_test_image_file():
    return 'tests/data/tensorplow/missing.jpg'


def test_load_image(test_image_file):
    image = tp.load_image(test_image_file)
    content = tp.run(image)
    assert content.shape == (256, 341, 3)


def test_load_non_existing_image(non_existing_test_image_file):
    with pytest.raises(FileNotFoundError):
        tp.load_image(non_existing_test_image_file)
