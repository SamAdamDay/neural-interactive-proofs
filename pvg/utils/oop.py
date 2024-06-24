"""Utilities to work with object-oriented programming (classes and such)."""


class classproperty:
    """A decorator to create a class property.

    This is a class property that can be accessed on the class itself, rather than on an
    instance of the class.

    Example
    -------
    >>> class MyClass:
    >>>     @classproperty
    >>>     def my_property(cls):
    >>>         return "my_property"

    Parameters
    ----------
    property_method : Callable
        The function to use as the class property.
    """

    def __init__(self, property_method):
        self.property_method = property_method

    def __get__(self, obj, owner):
        return self.property_method(owner)
