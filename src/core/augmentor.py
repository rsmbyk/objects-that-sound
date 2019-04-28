import numpy as np

from util import bit


class Aug:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class Augmentor:
    def __init__(self, *augmentors):
        if not all(map(lambda aug: isinstance(aug, Aug), augmentors)):
            raise TypeError('\'augmentors\' must be of type {}'.format(Aug))

        self.__augmentors = augmentors

    def __call__(self, item):
        augmented_items = list()

        for b in bit.bitstring(len(self.__augmentors)):
            x = item
            for i in range(len(self.__augmentors)):
                if b[i]:
                    x = self.__augmentors[i](x)
            augmented_items.append(np.array(x))

        return augmented_items

    def __len__(self):
        return pow(2, len(self.__augmentors))
