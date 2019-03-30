import json

import pytest

from core.ontology import RootEntry, Entry


@pytest.fixture
def test_root_entry_json():
    return {'id': 'root_id', 'name': 'Root Entry'}


@pytest.fixture
def test_entry_json():
    with open('tests/data/ontology/entry.json') as file:
        return json.load(file)


@pytest.fixture
def test_child_entry_json():
    with open('tests/data/ontology/child.json') as file:
        return json.load(file)


def test_root_entry_properties(test_root_entry_json):
    entry = RootEntry(**test_root_entry_json)
    assert entry.id == test_root_entry_json['id']
    assert entry.name == test_root_entry_json['name']


def test_slug_name(test_root_entry_json):
    entry = RootEntry(**test_root_entry_json)
    assert entry.slug_name == 'root-entry'


def test_identifiers_should_contains_root_identifier(test_root_entry_json):
    entry = RootEntry(**test_root_entry_json)
    assert entry.root_identifier in entry.identifiers


def test_initial_children(test_root_entry_json):
    entry = RootEntry(**test_root_entry_json)
    assert len(entry.children) == 0


def test_equality(test_root_entry_json):
    entry = RootEntry(**test_root_entry_json)
    assert entry == entry


def test_equality_with_different_instance(test_root_entry_json):
    entry1 = RootEntry(**test_root_entry_json)
    entry2 = RootEntry(**test_root_entry_json)
    assert entry1 != entry2


def test_equality_against_invalid_type(test_root_entry_json):
    entry = RootEntry(**test_root_entry_json)
    assert entry != 0


def test_equality_against_str_should_fails(test_root_entry_json):
    entry = RootEntry(**test_root_entry_json)
    assert entry != entry.id
    assert entry != entry.name
    assert entry != entry.slug_name


def test_string_representation_should_contains_id(test_root_entry_json):
    entry = RootEntry(**test_root_entry_json)
    assert entry.id in str(entry)


def test_wrap(test_root_entry_json, test_entry_json, test_child_entry_json):
    root_entry = RootEntry(**test_root_entry_json)
    entry = Entry(**test_entry_json)
    child = Entry(**test_child_entry_json)
    entry.link(child)
    root_entry.wrap(entry)
    assert entry in root_entry.children


# noinspection PyTypeChecker
def test_wrap_with_invalid_type(test_root_entry_json):
    root_entry = RootEntry(**test_root_entry_json)
    with pytest.raises(TypeError):
        root_entry.wrap(0)


def test_wrap_should_not_add_additional_children(test_root_entry_json, test_entry_json, test_child_entry_json):
    root_entry = RootEntry(**test_root_entry_json)
    entry = Entry(**test_entry_json)
    child = Entry(**test_child_entry_json)
    entry.link(child)
    root_entry.wrap(entry)
    assert len(root_entry.children) == 1


def test_wrap_should_add_child_ids(test_root_entry_json, test_entry_json):
    root_entry = RootEntry(**test_root_entry_json)
    entry = Entry(**test_entry_json)
    root_entry.wrap(entry)
    assert entry.id in root_entry.child_ids


def test_wrap_should_not_add_duplicate_child_ids(test_root_entry_json, test_entry_json):
    root_entry = RootEntry(**test_root_entry_json)
    entry = Entry(**test_entry_json)
    root_entry.wrap(entry)
    root_entry.wrap(entry)
    assert len(root_entry.child_ids) == 1


def test_items_should_contains_all_entries(test_root_entry_json, test_entry_json, test_child_entry_json):
    root_entry = RootEntry(**test_root_entry_json)
    entry = Entry(**test_entry_json)
    child = Entry(**test_child_entry_json)
    entry.link(child)
    root_entry.wrap(entry)
    items = entry.items()
    root_items = root_entry.items()
    assert all(map(lambda item: item in root_items, items))


def test_items_should_not_contains_additional_item(test_root_entry_json, test_entry_json, test_child_entry_json):
    root_entry = RootEntry(**test_root_entry_json)
    entry = Entry(**test_entry_json)
    child = Entry(**test_child_entry_json)
    entry.link(child)
    root_entry.wrap(entry)
    root_items = root_entry.items()
    assert len(root_items) == 2


def test_items_should_not_contains_root_id(test_root_entry_json, test_entry_json, test_child_entry_json):
    root_entry = RootEntry(**test_root_entry_json)
    entry = Entry(**test_entry_json)
    child = Entry(**test_child_entry_json)
    entry.link(child)
    root_entry.wrap(entry)
    root_items = root_entry.items()
    assert root_entry.id not in root_items


def test_root_entry_should_be_unable_to_be_linked(test_root_entry_json, test_entry_json):
    root_entry = RootEntry(**test_root_entry_json)
    entry = Entry(**test_entry_json)
    with pytest.raises(NotImplementedError):
        root_entry.link(entry)
