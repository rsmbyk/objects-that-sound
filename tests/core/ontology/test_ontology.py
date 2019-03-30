import pytest

from core.ontology import Ontology


@pytest.fixture
def test_ontology_file():
    return 'tests/data/ontology/ontology.json'


@pytest.fixture
def test_incomplete_ontology_file():
    return 'tests/data/ontology/incomplete.json'


@pytest.fixture
def test_invalid_ontology_file():
    return 'tests/data/ontology/invalid.json'


def test_initialize_ontology(test_ontology_file):
    ontology = Ontology(test_ontology_file)
    assert ontology.filename == test_ontology_file


def test_ontology_with_invalid_filename(test_invalid_ontology_file):
    with pytest.raises(FileNotFoundError):
        Ontology(test_invalid_ontology_file)


# noinspection PyTypeChecker
def test_ontology_with_invalid_filename_type():
    with pytest.raises(TypeError):
        Ontology(1)


def test_ontology_with_incomplete_ontology(test_incomplete_ontology_file):
    ontology = Ontology(test_incomplete_ontology_file)
    with pytest.raises(AssertionError):
        assert ontology.retrieve()


def test_retrieve(test_ontology_file):
    ontology = Ontology(test_ontology_file)
    retrieve = ontology.retrieve('human-sounds')
    assert 'human-sounds' in retrieve
    assert len(retrieve.items()) == 83


def test_retrieve_should_create_new_instance(test_ontology_file):
    ontology = Ontology(test_ontology_file)
    retrieve1 = ontology.retrieve()
    retrieve2 = ontology.retrieve()
    assert retrieve1 is not retrieve2


def test_retrieve_with_invalid_key(test_ontology_file):
    ontology = Ontology(test_ontology_file)
    with pytest.raises(KeyError):
        ontology.retrieve(0)


def test_retrieve_should_flatten_entry_in_same_hierarchy(test_ontology_file):
    ontology = Ontology(test_ontology_file)
    retrieve = ontology.retrieve('human-sounds', 'human-voice')
    assert len(retrieve.children) == 1
