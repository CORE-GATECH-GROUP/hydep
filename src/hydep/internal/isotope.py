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
    """Representation of a single isotope

    It is intended that a single instance be used to represent
    a unique isotope, e.g. all occurances of U235 should use
    the same :class:`Isotope` instance. This reduces memory
    consumption, as these classes are really data classes,
    useful for reactions only.

    Parameters
    ----------
    name : str
        GND-name of this isotope, e.g. ``"U235"``,
        ``"Am242_m1"``, etc.
    z : Union[int, hydep.internal.ZaiTuple]
        Isotope numeric identifier. If ``a`` and ``i`` are
        not given, then this must be a complete identifier, e.g.
        one of the following:

        * int - Single integer representing the ZAI identifier,
          e.g. ``922350``
        * :class:`hydep.internal.ZaiTuple` -
          triplet of ``(z, a, i)``, e.g. ``(92, 235, 0)``
    a : Optional[int]
        Integer reflecting the number of neutrons and protons
        in the isotope
    i : Optional[int]
        Integer reflecting the metastable state of the isotope

    Attributes
    ----------
    z : int
        Number of protons
    a : int
        Number of neutrons
    i : int
        Metastable state
    zai : int
        Full ``zzaaai`` number - 922350 for U235
    triplet : hydep.internal.ZaiTuple
        Iterable with ``(z, a, i)`` attributes. The main test for
        equality and sorting
    name : str
        Name of this isotope
    decayModes : set of DecayTuple
        If this isotope undergoes spontaneous decay, each entry
        describes a mechanism, target, and branching ratio for
        a unique mode
    decayConstant : Union[float, None]
        If this isotope decays, this quantity describes the decay
        constant, or :math:`ln(2)/t_{1/2}``
    reactions : set of ReactionTuple
        Iterable describing the various neutron-induced
        transmutation reactions this isotope can experience
    fissionYields : hydep.internal.FissionYieldDistribution or None
        If this isotope has fission yields, they will populate
        a distribution here. Can be treated as a mapping
        ``{energy : {zai : float}}`` for various product isotopes
        at various neutron energies

    """

    # TODO Find existing isotopes with __new__?
    # TODO Make this a dataclass? Python >= 3.7
    __slots__ = ("_name", "_zai", "decayModes", "reactions",
                 "decayConstant", "fissionYields")

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

        self.decayConstant = None
        self.decayModes = set()
        self.reactions = set()
        self.fissionYields = None

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
    """Return an isotope given a name and/or its ZAI

    Either ``name`` or ``zai`` must be given. If both are
    given, then ``zai`` will be preferred. It is not
    guaranteed that the :attr:`Isotope.name` will always
    match if GND names, e.g. ``"U235"``, ``"Am241_m1"``
    are not used.

    Parameters
    ----------
    name : Optional[str]
        Name of this isotope. If given as the only argument,
        then will be parsed as a GND name of the form
        ``(atomic symbol)(number of protons +
        neutrons)[_m(metastable state)]``
    zai : Optional[Union[int, Iterable[int]]]
        Isotope numeric identifier. Can be a single integer
        representing ``zzaaai`` value, e.g. 922350 for U235,
        or an iterable of integers ``(z, a, i)``

    Returns
    -------
    hydep.internal.Isotope
    """
    assert (name is not None) != (zai is not None)

    if name is not None:
        zai = getZaiFromName(name)
    else:
        zai = parseZai(zai)

    isotope = _ISOTOPES.get(zai)
    if isotope is not None:
        return isotope

    if name is None:
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
