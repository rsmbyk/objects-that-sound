import os
import shutil
from contextlib import contextmanager


@contextmanager
def temp_dir(path):
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path)


@contextmanager
def temp_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)
    yield len(paths)
    for path in paths:
        shutil.rmtree(path)


@contextmanager
def temp_copy(file, to):
    yield shutil.copy(file, to)
    os.remove(os.path.join(to, os.path.basename(file)))


@contextmanager
def temp_copy_to_multiple_dirs(file, *paths):
    for path in paths:
        shutil.copy(file, path)
    yield len(paths)
    for path in paths:
        os.remove(os.path.join(path, os.path.basename(file)))


@contextmanager
def temp_copy_file(file, to):
    yield shutil.copyfile(file, to)
    os.remove(to)


@contextmanager
def temp_copy_file_to_multiple_dirs(file, *paths):
    for path in paths:
        shutil.copyfile(file, path)
    yield len(paths)
    for path in paths:
        os.remove(path)


@contextmanager
def temp_move(src, dst):
    os.rename(src, dst)
    yield dst
    os.rename(dst, src)
