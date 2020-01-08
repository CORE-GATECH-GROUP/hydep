"""Small helpers that don't necessitate a full file"""
from collections import namedtuple
from collections.abc import Mapping
from configparser import ConfigParser
from functools import wraps
from pathlib import Path

__all__ = ("Boundaries", "configmethod", "CompBundle")

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


def configmethod(m):
    """Decorator for assisting with configuration

    .. currentmodule:: configparser

    Acts on a method intended to configure some aspect of the
    project. Works assuming the intended method has the call
    signature ``MyClass.config(self, config, ...)``, and performs
    the following actions before returning to the method

    1. If ``config`` is a :class:`ConfigParser`,
       then take no action and pass ``config`` directly
    2. If ``config`` is a string, :class:`pathlib.Path`, or
       :class:`collections.abc.Mapping`, create a new
       parser, load in the data with
       :meth:`ConfigParser.read_file` or
       :meth:`ConfigParser.read_dict`. This new parser
       is passed to the method

    This way, methods wrapped with this decorator
    can assume that the config object passed will
    always be a :class:`ConfigParser`.

    Raises
    ------
    TypeError
        If the anticipated config object is not one of
        the aforementioned types

    """
    @wraps(m)
    def wrapper(s, cfg, *args, **kwargs):
        if isinstance(cfg, ConfigParser):
            return m(s, cfg, *args, **kwargs)
        parser = ConfigParser()
        if isinstance(cfg, Mapping):
            parser.read_dict(cfg)
        elif isinstance(cfg, str):
            parser.read_file(open(cfg, "r"))
        elif isinstance(cfg, Path):
            # VER python >= 3.6 config knows how to handle Path
            parser.read_file(cfg.open("r"))
        else:
            raise TypeError(type(cfg))
        return m(s, parser, *args, **kwargs)
    return wrapper


CompBundle = namedtuple("CompBundle", "zai densities")
CompBundle.__doc__ = """Updated compositions following a depletion event

Parameters
----------
zai : Iterable[int]
    Ordering of isotopics
densities : Iterable[Iterable[float]]
    Iterable of material compositions such that ``densities[i][j]``
    is the atom density of isotope ``zai[j]`` for burnable
    material ``i``

Attributes
----------
zai : Iterable[int]
    Ordering of isotopics
densities : Iterable[Iterable[float]]
    Iterable of material compositions such that ``densities[i][j]``
    is the atom density of isotope ``zai[j]`` for burnable
    material ``i``

"""
