"""Small helpers that don't necessitate a full file"""
from collections import namedtuple
from collections.abc import Mapping, Sequence
from configparser import ConfigParser
from functools import wraps
import itertools
from pathlib import Path
import numbers
import typing

import numpy

__all__ = ("Boundaries", "configmethod", "CompBundle", "compBundleFromMaterials")

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


CompBundle = namedtuple("CompBundle", "isotopes densities")
CompBundle.__doc__ = """Updated compositions following a depletion event

Parameters
----------
isotope : Iterable[hydep.internal.Isotope]
    Ordering of isotopics
densities : Iterable[Iterable[float]]
    Iterable of material compositions such that ``densities[i][j]``
    is the atom density [#/b-cm] of isotope ``isotope[j]`` for burnable
    material ``i``

Attributes
----------
isotope : Iterable[hydep.internal.Isotope]
    Ordering of isotopics
densities : Iterable[Iterable[float]]
    Iterable of material compositions such that ``densities[i][j]``
    is the atom density [#/b-cm] of isotope ``isotope[j]`` for burnable
    material ``i``

"""


class FakeSequence(Sequence):
    """Emulate an iterable of repeated data

    Create a proxy-iterable that stores one entry, but represents
    an iterable of multiple entries. Like ``[data] * N`` but
    better for iteration.

    .. note::

        Calls to :meth:`index`, :meth:`count`, and
        :meth:`__contains__` require comparisons
        against the stored data, e.g. ``value == data``
        which can be problematic for some cases. This behavior
        is discouraged and may be prone to errors.

    .. warning::

        Manipulations during iteration will be propagated
        to all future events, as the underlying data is shared across
        all steps.

    Parameters
    ----------
    data : object
        Data to be stored
    counts : int
        Number of entries in the list to emulate

    Examples
    --------
    >>> l = FakeSequence([1, 2, 3], 5)
    >>> len(l)
    5
    >>> l[0]
    [1, 2, 3]
    >>> l[-1]
    [1, 2, 3]
    >>> for item in l:
    ...     print(item)
    [1, 2, 3]
    [1, 2, 3]
    [1, 2, 3]
    [1, 2, 3]
    [1, 2, 3]
    >>> [1, 2, 3] in l
    True
    >>> l.index([1, 2, 3])
    0

    """

    __slots__ = ("_data", "_n")

    def __init__(self, data: typing.Any, counts: int):
        self._data = data
        self._n = counts

    def __getitem__(self, index: int) -> typing.Any:
        if index < 0:
            index = self._n - 1 + index
        if index < self._n:
            return self._data
        raise IndexError("Index out of range")

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        return itertools.repeat(self._data, self._n)

    def __reversed__(self):
        return iter(self)

    def __contains__(self, value) -> bool:
        return value is self._data or value == self._data

    def index(self, value) -> int:
        if value in self:
            return 0
        raise IndexError(value)

    def count(self, value) -> int:
        return len(self) if value in self else 0


def compBundleFromMaterials(
        materials: typing.Sequence[typing.Mapping[typing.Any, float]],
        threshold: typing.Optional[float]=0) -> CompBundle:
    """Create a :class:`CompBundle` from material definitions

    Parameters
    ----------
    materials : sequence of :class:`hydep.Material`
        Materials containing atom densities [#/b/cm] that will
        fill up :attr:`CompBundle.densities`. A sorted, common set of
        isotopes will be placed in :attr:`CompBundle.isotopes`
    threshold : float, optional
        Minimum density [#/b/cm] for isotopes to be included.
        If no material contains a given isotope with a density
        greater than ``threshold``, that isotope and corresponding
        densities will be removed from the bundle.

    Returns
    -------
    CompBundle

    """

    if not isinstance(threshold, numbers.Real):
        raise TypeError(type(threshold))

    isotopes = set()
    for m in materials:
        isotopes.update(m)

    isotopes = sorted(isotopes)

    densities = numpy.empty((len(materials), len(isotopes)))

    for matix, mat in enumerate(materials):
        for isox, isotope in enumerate(isotopes):
            densities[matix, isox] = mat.get(isotope, 0.0)

    if not threshold:
        return CompBundle(tuple(isotopes), densities)

    isoFlags = (densities > threshold).any(axis=0)

    return CompBundle(
        tuple(isotopes[ix] for ix, f in enumerate(isoFlags) if f),
        densities[:, isoFlags].copy(),
    )
