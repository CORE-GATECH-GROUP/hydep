from hydep.internal.registry import register, unregister, get


def test_register():
    """Test the registry interface"""
    class Reg:
        def __init__(self):
            self._id = register(self.__class__)

        @property
        def id(self):
            return self._id

        def __del__(self):
            unregister(self.__class__)

    assert get(Reg) == 0
    r = Reg()
    assert r.id == 1
    assert get(Reg) == 1
    assert get(r.__class__) == 1
    del r
    assert get(Reg) == 0
    r = Reg()
    n = Reg()
    assert r.id == 1
    assert n.id == 2
    assert get(Reg) == 2
    del n
    assert get(Reg) == 1
