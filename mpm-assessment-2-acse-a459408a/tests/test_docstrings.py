import doctest
import mpm_la as m


def test_doctest_all():
    obj = m.functions
    fail = doctest.testmod(obj)[0]
    assert (True if fail == 0 else False)
