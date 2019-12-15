"""Small helpers that don't necessitate a full file"""
from collections import namedtuple

__all__ = ("Boundaries", )

Boundaries = namedtuple("Boundaries", "x y z")
Boundaries.__doc__ = """Representation of spatial domains

For all parameters, a value of ``None`` indicates an
unbounded domain in a specific direction. The other
alternative is to use ``numpy.inf`` for one or both
items.

Parameters
----------
x : iterable of float or None
    Lower and upper boundaries in the x direction.
y : iterable of float or None
    Lower and upper boundaries in the y direction.
z : iterable of float or None
    Lower and upper boundaries in the z direction.

Attributes
----------
x : iterable of float or None
    Lower and upper boundaries in the x direction.
y : iterable of float or None
    Lower and upper boundaries in the y direction.
z : iterable of float or None
    Lower and upper boundaries in the z direction.

Examples
--------
>>> b = Boundaries((-10.21, 10.21), (-10.21, 10.21), None)
>>> b.x
(-10.21, 10.21)
>>> b.z is None
True

"""
