import pytest
from util.threading import fork


@pytest.fixture
def target():
    def thread_function(test_list):
        test_list.append(1)

    return thread_function


@pytest.mark.parametrize('count', [1, 2, 3, 5])
def test_fork(target, count):
    test_list = list()
    args = ((test_list,) for i in range(count))
    fork(count, target, *args)
    assert len(test_list) == count


def test_fork_with_different_arguments_length(target):
    test_list = list()
    args = ((test_list,) for i in range(2))
    with pytest.raises(ValueError):
        fork(3, target, *args)
