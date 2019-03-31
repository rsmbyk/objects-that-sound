import json
import os

import pytest

from core.ontology import Entry


@pytest.fixture
def test_root_dir():
    return 'tests/.temp/ontology'


@pytest.fixture
def test_entry_json():
    with open('tests/data/ontology/entry.json') as file:
        return json.load(file)


@pytest.fixture
def test_child_entry_json():
    with open('tests/data/ontology/child.json') as file:
        return json.load(file)


def test_entry_properties(test_entry_json):
    entry = Entry(None, **test_entry_json)
    assert entry.root_dir is None
    assert entry.id == test_entry_json['id']
    assert entry.name == test_entry_json['name']
    assert entry.description == test_entry_json['description']
    assert entry.citation_uri == test_entry_json['citation_uri']
    assert entry.positive_examples == test_entry_json['positive_examples']
    assert entry.child_ids == test_entry_json['child_ids']
    assert entry.restrictions == test_entry_json['restrictions']


def test_slug_name(test_entry_json):
    entry = Entry(None, **test_entry_json)
    assert entry.slug_name == 'human-sounds'


def test_proper_name(test_entry_json):
    entry = Entry(None, **test_entry_json)
    assert '.' not in entry.proper_name


def test_identifiers(test_entry_json):
    entry = Entry(None, **test_entry_json)
    assert entry.identifiers == (entry.id, entry.name, entry.slug_name)


def test_initial_parents(test_entry_json):
    entry = Entry(None, **test_entry_json)
    assert len(entry.parents) == 0


def test_initial_children(test_entry_json):
    entry = Entry(None, **test_entry_json)
    assert len(entry.children) == 0


def test_equality(test_entry_json):
    entry1 = Entry(None, **test_entry_json)
    entry2 = Entry(None, **test_entry_json)
    assert entry1 == entry2


def test_equality_against_str(test_entry_json):
    entry = Entry(None, **test_entry_json)
    assert entry == '/m/0dgw9r'
    assert entry == 'Human sounds'
    assert entry == 'human-sounds'


def test_equality_with_invalid_type(test_entry_json):
    entry = Entry(None, **test_entry_json)
    assert entry != 1


def test_string_representation_should_contains_id(test_entry_json):
    entry = Entry(None, **test_entry_json)
    assert entry.id in str(entry)


def test_items_should_contains_all_entries(test_entry_json, test_child_entry_json):
    entry = Entry(None, **test_entry_json)
    child = Entry(None, **test_child_entry_json)
    entry.link(child)
    items = entry.items()
    assert entry.id in items
    assert child.id in items


def test_items_should_not_contains_additional_item(test_entry_json, test_child_entry_json):
    entry = Entry(None, **test_entry_json)
    child = Entry(None, **test_child_entry_json)
    entry.link(child)
    items = entry.items()
    assert len(items) == 2


def test_link(test_entry_json, test_child_entry_json):
    entry = Entry(None, **test_entry_json)
    child = Entry(None, **test_child_entry_json)
    entry.link(child)
    assert len(entry.children) == 1
    assert len(child.parents) == 1
    assert child in entry.children
    assert entry in child.parents


# noinspection PyTypeChecker
def test_link_with_invalid_type(test_entry_json):
    entry = Entry(None, **test_entry_json)
    with pytest.raises(TypeError):
        entry.link('child')


def test_link_with_intruder_child(test_entry_json):
    entry = Entry(None, **test_entry_json)
    child = Entry(None, 'invalid_id', 'Test Child')
    with pytest.raises(ValueError):
        entry.link(child)


def test_link_already_linked_child(test_entry_json, test_child_entry_json):
    entry = Entry(None, **test_entry_json)
    child = Entry(None, **test_child_entry_json)
    entry.link(child)
    with pytest.raises(ValueError):
        entry.link(child)


def test_contains_with_str(test_entry_json, test_child_entry_json):
    entry = Entry(None, **test_entry_json)
    child = Entry(None, **test_child_entry_json)
    entry.link(child)
    assert child.id in entry
    assert 'invalid_id' not in entry


def test_contains_with_entry(test_entry_json, test_child_entry_json):
    entry = Entry(None, **test_entry_json)
    child = Entry(None, **test_child_entry_json)
    invalid = Entry(None, 'invalid_id', 'Test Child')
    entry.link(child)
    assert child in entry
    assert invalid not in entry


def test_contains_with_invalid_type(test_entry_json):
    entry = Entry(None, **test_entry_json)
    assert 1 not in entry


def test_paths(test_entry_json, test_child_entry_json):
    entry = Entry(None, **test_entry_json)
    child = Entry(None, **test_child_entry_json)
    entry.link(child)
    assert os.path.join(entry.proper_name, child.proper_name) in child.paths


def test_dirs(test_root_dir, test_entry_json, test_child_entry_json):
    entry = Entry(root_dir=test_root_dir, **test_entry_json)
    child = Entry(root_dir=test_root_dir, **test_child_entry_json)
    entry.link(child)
    assert os.path.join(test_root_dir, entry.proper_name, child.proper_name) in child.dirs


def test_getitem(test_entry_json, test_child_entry_json):
    entry = Entry(None, **test_entry_json)
    child = Entry(None, **test_child_entry_json)
    entry.link(child)
    assert entry[child.id] == child


def test_getitem_with_invalid_key(test_entry_json):
    entry = Entry(None, **test_entry_json)
    with pytest.raises(KeyError):
        assert entry['invalid_id']


def test_getitem_with_invalid_type_key(test_entry_json):
    entry = Entry(None, **test_entry_json)
    with pytest.raises(KeyError):
        assert entry[1]
