from .singleton import Singleton

def test_identity():
    class Foo(metaclass=Singleton):
        def __init__(self, a):
            self.a = a
        pass

    foo1 = Foo(1)
    assert foo1.a == 1

    foo2 = Foo()
    foo2.a = 2
    assert foo1.a == 2

    # assert foo1 is foo2
