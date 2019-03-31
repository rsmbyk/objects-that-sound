import os
import pandas as pd
import re
from typing import List, Dict


class Entry:
    # noinspection PyShadowingBuiltins
    def __init__(self, root_dir, id, name,
                 description=None, citation_uri=None,
                 positive_examples=None, child_ids=None, restrictions=None):

        self.root_dir: str = root_dir
        self.id: str = id
        self.name: str = name
        self.description: str = description or ''
        self.citation_uri: str = citation_uri or ''
        self.positive_examples: List[str] = positive_examples or []
        self.child_ids: List[str] = child_ids or []
        self.restrictions: List[str] = restrictions or []

        self._children: List[Entry] = list()
        self._parents: List[Entry] = list()
        self._keys_cache = dict()

    @property
    def slug_name(self):
        return '-'.join(map(str.lower, re.split(r'\W+', self.name)))

    @property
    def proper_name(self):
        return self.name.strip('.')

    @property
    def identifiers(self):
        return self.id, self.name, self.slug_name

    @property
    def children(self):
        return self._children

    @property
    def parents(self):
        return self._parents

    @property
    def paths(self):
        if not self.parents:
            return [self.proper_name]

        return [os.path.join(path, self.proper_name)  # join
                for parent in self.parents            # list of parent
                for path in parent.paths]             # list of path in each parent

    @property
    def dirs(self):
        return list(map(lambda p: os.path.join(self.root_dir, p), self.paths))

    def items(self):
        """ return flat list of all children. """
        return list(set([self.id] + [item                          # item
                                     for child in self.children    # list of child
                                     for item in child.items()]))  # list of item in each child

    def link(self, *children: 'Entry'):
        if not all(map(lambda x: isinstance(x, Entry), children)):
            raise TypeError('\'children\' must be of type {}'.format(Entry))

        intruder = list(filter(lambda x: x.id not in self.child_ids, children))
        if any(intruder):
            raise ValueError('Invalid child ids (in {}): {}'.format(self, intruder))

        existing = list(filter(lambda x: x.id in self.children, children))
        if any(existing):
            raise ValueError('These entries has already been linked: {}'.format(existing))

        self.children.extend(children)
        for child in children:
            child.parents.append(self)

    def __getitem__(self, item):
        """
        Retrieve entry with provided `item` identifier from any of `id`, `name`, `slug_name` or `proper_name`.
        """
        if isinstance(item, str):
            if item == self:
                return self

            for child in self.children:
                if item in child:
                    return child[item]

            raise KeyError(item)

        raise KeyError('Can only retrieve entry by its identifiers')

    def __contains__(self, item):
        """
        Check if given item(s) or identifier(s) are exist in this ontology hierarchy.
        """
        if not isinstance(item, (list, tuple)):
            item = [item]

        def contains(key):
            if isinstance(key, Entry):
                return key.identifiers in self

            if not isinstance(key, str):
                return False

            if key not in self._keys_cache:
                self._keys_cache[key] =\
                    self == key or any(map(lambda child: key in child, self.children))

            return self._keys_cache[key]

        return all(map(contains, item))

    def __eq__(self, other):
        if isinstance(other, str):
            return other in self.identifiers
        if isinstance(other, Entry):
            return self.identifiers == other.identifiers
        return False

    def __str__(self):
        return '({}: {})'.format(self.id, self.slug_name)


class RootEntry(Entry):
    root_identifier = 0

    # noinspection PyShadowingBuiltins
    def __init__(self, id, name):
        super().__init__(None, id, name)
        self.root_identifier = RootEntry.root_identifier
        RootEntry.root_identifier += 1

    @property
    def identifiers(self):
        return (*super().identifiers, self.root_identifier)

    def link(self, *childre):
        raise NotImplementedError('Root Entry can\'t be linked with another Entry'
                                  'Use wrap instead')

    def wrap(self, *children: Entry):
        if not all(map(lambda x: isinstance(x, Entry), children)):
            raise TypeError('\'children\' must be of type {}'.format(Entry))

        self._children = children
        self.child_ids = list(set(self.child_ids + list(map(lambda x: x.id, children))))

    def items(self):
        items_ = super().items()
        # exclude self.id from items
        items_.remove(self.id)
        return items_

    def __eq__(self, other):
        if isinstance(other, Entry):
            return super().__eq__(other)
        return False


class Ontology:
    def __init__(self, filename, root_dir):
        if not os.path.exists(filename):
            raise FileNotFoundError('FILE ({})'.format(filename))

        if not isinstance(filename, str):
            raise TypeError('FILENAME cannot be of type {}', type(filename))

        self.__root__ = None
        self.__root_dir = root_dir
        self.__filename = filename

    @property
    def filename(self):
        return self.__filename

    @property
    def root_dir(self):
        return self.__root_dir

    @property
    def __root(self):
        if not self.__root__:
            ontology = pd.read_json(self.__filename).set_index('id')
            labels: Dict[str, Entry] =\
                {mid: Entry(root_dir=self.root_dir,
                            id=mid,
                            **row.to_dict())
                 for mid, row
                 in ontology.iterrows()}

            for label in labels.values():
                missing_ids = list(filter(lambda x: x not in labels, label.child_ids))
                if missing_ids:
                    raise AssertionError(
                        'Can\'t build ontology. '
                        'Missing child id(s) {} for {}'.format(missing_ids, label.id))
                label.link(*map(lambda x: labels[x], label.child_ids))

            root_labels = {mid: label for mid, label in labels.items() if not label.parents}
            self.__root__ = RootEntry('/r/00t', 'AudioSet Ontology')
            self.__root__.wrap(*root_labels.values())
        return self.__root__

    def retrieve(self, *labels):
        if not labels:
            labels = ['*']
            result = self.__root.children
        else:
            result = [self.__root[label] for label in labels]

        result = {label.id: label for label in result if label}

        # check if any Label superset another
        result_keys = list(result.keys())
        result_copy = [set(label.items()) for label in result.values()]
        for it1 in result_copy:
            for i, it2 in enumerate(result_copy):
                if it1 != it2 and it1.issuperset(it2) and result_keys[i] in result:
                    del result[result_keys[i]]

        if len(labels) == 1:
            labels = labels[0]
        labels = str(labels)

        _root = RootEntry('/r/etr1eve', labels)
        _root.wrap(*result.values())
        return _root
