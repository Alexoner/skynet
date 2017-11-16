class Singleton(type):
    """
    Define an Instance operation that lets clients access its unique
    instance.
    """

    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class MyClass(metaclass=Singleton):
    """
    Example class.
    """

    pass


# class Singleton(object):
    # _instance = None

    # def __new__(cls, *args, **kwargs):
        # if not isinstance(cls._instance, cls):
            # print("Constructing singleton for", cls, 'with parameters', args, kwargs)
            # # cls._instance = object.__new__(cls, *args, **kwargs)
            # cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        # return cls._instance

# class Singleton(object):
  # _instance = None
  # def __new__(cls, *args, **kwargs):
    # if not isinstance(cls._instance, cls):
        # cls._instance = object.__new__(cls, *args, **kwargs)
    # return cls._instance
