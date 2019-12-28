"""Functions for registering items that must be counted

It is often important to represent a specific material or geometric
item (pin cell, lattice, etc.) with a unique identifier. Most codes
require this to be a positive integer, e.g. universe id in a Monte
Carlo code. One could use ``hex(id(obj))`` to obtain a unique number,
but this is not reproducible, a desired attribute for testing.

This module provides functions for producing ids for classes,
inspecting the registry, and removing items. These could and should
be used in object construction and destruction to ensure a correct
registry.

>>> class RegDemo:
...     def __init__(self):
...         self.id = register(self.__class__)
...     def __del__(self):
...         unregister(self.__class__)
>>> r = RegDemo()
>>> r.id
1
>>> n = RegDemo()
>>> n.id
2
>>> del n
>>> m = RegDemo()
>>> m.id
2

.. warning::

    The registry has no knowledge of copies created by
    :func:`copy.copy` or :func:`copy.deepcopy`. This
    can destroy the guarantee of uniqueness.

>>> import copy
>>> rc = copy.copy(r)
>>> rc.id == r.id
True
>>> rc is r
False
"""

__all__ = ("register", "unregister", "get")

__registry = {}


def get(klass) -> int:
    """Return the value in the registry for a specific key

    Parameters
    ----------
    klass : type
        Potentially un-registered class to be examined

    Returns
    -------
    int
        Number of registered items in existence. Guaranteed to
        be non-negative.

    """
    return __registry.get(klass, 0)


def register(klass) -> int:
    """Register a class

    Increments the counter for this class and
    returns a positive integer representing
    the number of items now in the registry
    for this class.

    Parameters
    ----------
    klass : type
        Class to be registered

    Returns
    -------
    int
        Positive integer ( >= 1) of items
        in the registry for this class

    """
    current = get(klass)
    __registry[klass] = current + 1
    return current + 1


def unregister(klass):
    """Remove an occurrence in the registry

    This routine should be called whenever an instance of a registered
    class is destroyed.

    Parameters
    ----------
    klass : type
        Class of removed instance
    """
    count = __registry.get(klass)
    if count is None:
        return
    if count > 1:
        __registry[klass] -= 1
    else:
        __registry.pop(klass)
