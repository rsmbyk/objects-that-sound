import util.tensorplow as tp


@tp.get_ops(tp.LoadImage)
def func(ops):
    return ops.__class__.__name__


@tp.get_ops(tp.EncodePNG)
def func_ops(ops):
    return ops


def test_get_ops():
    func()
    assert tp.LoadImage in tp.tp_ops


def test_function_result_not_modified():
    assert func() == 'LoadImage'


def test_ops_should_be_same():
    assert func_ops() is func_ops()
