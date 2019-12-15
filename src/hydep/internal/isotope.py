import re
from collections import namedtuple
from collections.abc import Iterable
import numbers

from .symbols import NUMBERS, SYMBOLS

__all__ = [
    "ZaiTuple", "getIsotope", "getZaiFromName", "Isotope", "ReactionTuple",
    "DecayTuple", "parseZai",
]

ZaiTuple = namedtuple("ZaiTuple", "z a i")
# Inspired by OpenMC depletion module
ReactionTuple = namedtuple("ReactionTuple", ["mt", "target", "branch", "Q"])
DecayTuple = namedtuple("DecayTuple", ["target", "type", "branch"])

# TODO weakref?
_ISOTOPES = {}

_GND_REG = re.compile(r"([A-z]+)([0-9]+)[_m]{0,2}([0-9]*)")


class Isotope:

    # TODO Find existing isotopes with __new__?
    # TODO Make this a dataclass? Python >= 3.7
    __slots__ = ("_name", "_zai", "decayModes", "reactions", "decayConstant")

    def __init__(self, name, z, a=None, i=None):
        self._name = name
        if isinstance(z, ZaiTuple):
            self._zai = z
        else:
            if not isinstance(z, numbers.Integral):
                raise TypeError("z: {}".format(type(z)))
            elif z < 0:
                raise ValueError("z: {}".format(z))
            if not isinstance(a, numbers.Integral):
                raise TypeError("a: {}".format(type(a)))
            elif a < 0:
                raise ValueError("a: {}".format(a))
            if i is None:
                i = 0
            elif not isinstance(i, numbers.Integral):
                raise TypeError("i: {}".format(type(i)))
            elif i < 0:
                raise ValueError("i: {}".format(i))
            self._zai = ZaiTuple(z, a, i)

        self.decayModes = set()
        self.reactions = set()

    def __le__(self, other):
        if isinstance(other, self.__class__):
            return self._zai <= other._zai
        elif isinstance(other, ZaiTuple):
            return self._zai <= other
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self._zai < other._zai
        elif isinstance(other, ZaiTuple):
            return self._zai < other
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._zai == other._zai
        elif isinstance(other, ZaiTuple):
            return self._zai == other
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, self.__class__):
            return self._zai >= other._zai
        elif isinstance(other, ZaiTuple):
            return self._zai >= other
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self._zai >= other._zai
        elif isinstance(other, ZaiTuple):
            return self._zai >= other
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self._zai)

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, self._name)

    @property
    def zai(self):
        return self._zai.z * 10000 + self._zai.a * 10 + self._zai.i

    @property
    def triplet(self):
        return self._zai

    @property
    def z(self):
        return self._zai.z

    @property
    def a(self):
        return self._zai.a

    @property
    def i(self):
        return self._zai.i

    @property
    def name(self):
        return self._name


def getIsotope(name=None, zai=None):
    assert (name is not None) != (zai is not None)

    if name is not None:
        zai = getZaiFromName(name)
    else:
        zai = parseZai(zai)

    isotope = _ISOTOPES.get(zai)
    if isotope is not None:
        return isotope

    name = SYMBOLS[zai.z] + str(zai.a)
    if zai.i:
        name += "_m{}".format(zai.i)

    isotope = Isotope(name, zai)
    _ISOTOPES[zai] = isotope
    return isotope


def parseZai(zai):
    if isinstance(zai, ZaiTuple):
        return zai
    if isinstance(zai, numbers.Integral):
        za, i, = divmod(zai, 10)
        return ZaiTuple(*divmod(za, 1000), i)
    if isinstance(zai, Iterable):
        assert all(isinstance(x, numbers.Integral) for x in zai)
        if len(zai) == 2:
            return ZaiTuple(*zai, 0)
        elif len(zai) == 3:
            return ZaiTuple(*zai)
        else:
            raise ValueError("Expected two or three items in ZAI, not {}".format(zai))
    raise TypeError(
        "Unsupported type {} cannot be converted to ZAI. Expected {}, integer, "
        "or sequence or (Z, A, [I])".format(type(zai), ZaiTuple))

def getZaiFromName(name):
    match = _GND_REG.match(name)
    if match is None:
        raise ValueError(name)
    name, a, i = match.groups()
    z = NUMBERS[name]
    a = int(a)
    i = int(i) if i else 0
    return ZaiTuple(z, a, i)
