import warnings
import numbers
from collections.abc import Mapping, Iterable
import typing

import numpy

from hydep.typed import BoundedTyped
from hydep.internal import Isotope, ZaiTuple, getIsotope
from hydep.internal.registry import register, unregister

IsotopeLike = typing.Union[int, str, Isotope]


__all__ = ["Material", "BurnableMaterial"]


class Material(dict):
    r"""Basic material comprised of nuclides and atom densities

    Implements a Mapping interface

    Parameters
    ----------
    name : str
        Name of this material
    adens : numbers.Real, optional
        Atomic density [atoms/b/cm] of this material. Mutually exclusive
        with ``mdens``
    mdens : numbers.Real, optional
        Mass density [g/cm^3] of this material. Mutually exclusive with
        ``adens``
    temperature : numbers.Real, optional
        Temperature [K] of this material
    volume : numbers.Real, optional
        Volume [cm^3] of this material
    **nucs :
        Additional keyword arguments are treated as nuclides
        and corresponding atomic densities.

    Attributes
    ----------
    name : str
        Name of this material
    adens : numbers.Real or None
        Atomic density [atoms/b/cm] of this material. Mutually exclusive
        with :attr:`mdens`
    mdens : numbers.Real or None
        Mass density [g/cm^3] of this material. Mutually exclusive with
        :attr:`adens`
    temperature : numbers.Real or None
        Temperature [K]
    volume : numbers.Real or None
        Volume [cm^3]
    sab : set of str
        Names of :math:`S(\alpha,\beta)` thermal scattering tables to
        be included for this material. Should be added through
        :meth:`addSAlphaBeta`
    id : int
        A unique positive identifier for this material.

    Examples
    --------
    >>> mat = Material("fuel", U235=1.0)
    >>> mat["U235"]
    1.0

    """
    # TODO Something for atom fractions
    _adens = BoundedTyped("adens", numbers.Real, gt=0.0, allowNone=True)
    _mdens = BoundedTyped("mdens", numbers.Real, gt=0.0, allowNone=True)
    temperature = BoundedTyped("temperature", numbers.Real, gt=0.0, allowNone=True)
    volume = BoundedTyped("volume", numbers.Real, gt=0.0, allowNone=True)

    def __init__(
        self, name, adens=None, mdens=None, temperature=None, volume=None, **nucs
    ):
        if (adens is not None) and (mdens is not None):
            raise ValueError(
                "Cannot set both adens and mdens for {} {}".format(
                    self.__class__.__name__, name
                )
            )
        self.name = name
        self._adens = adens
        self._mdens = mdens
        self.temperature = temperature
        self.volume = volume
        self.sab = set()
        super().__init__()
        self._id = register(Material)
        self.update(nucs)

    def __del__(self):
        unregister(Material)

    @property
    def id(self):
        return self._id

    @property
    def mdens(self):
        return self._mdens

    @mdens.setter
    def mdens(self, value):
        if value is None:
            self._mdens = None
        elif self._adens is not None:
            raise AttributeError("Cannot set both atomic and mass densities")
        else:
            self._mdens = value

    @property
    def adens(self):
        return self._adens

    @adens.setter
    def adens(self, value):
        if value is None:
            self._adens = None
        elif self._mdens is not None:
            raise AttributeError("Cannot set both atomic and mass densities")
        else:
            self._adens = value

    def update(self, other):
        if not isinstance(other, Mapping):
            other = dict(other)
        for key, value in other.items():
            self[key] = value

    @staticmethod
    def _getIsotopeFromKey(key):
        if isinstance(key, Isotope):
            return key
        if isinstance(key, str):
            return getIsotope(name=key)
        if isinstance(key, (Iterable, ZaiTuple, numbers.Integral)):
            return getIsotope(zai=key)
        raise TypeError(
            "Keys should be {}, {}, iterables, integers, or strings, "
            "not {}".format(Isotope.__class__.__name__, ZaiTuple, type(key))
        )

    def __setitem__(self, key: IsotopeLike, value: float):
        """Update or set atom density for a specific isotope

        Parameters
        ----------
        key : str or int or hydep.internal.Isotope
            Either isotope name, ZAI, or internal representation of an isotope
        value : float
            Atom density [atoms/b/cm] for this istope

        Examples
        --------
        >>> w = Material("water", mdens=9.75)
        >>> w["H1"] = 0.047
        >>> w["H1"]
        0.047
        >>> w[10010]
        0.047

        """
        assert isinstance(value, numbers.Real)
        assert value > 0
        super().__setitem__(self._getIsotopeFromKey(key), value)

    def __getitem__(self, key: IsotopeLike) -> float:
        """Return atom density for a specific isotope

        Parameters
        ----------
        key : str or int or hydep.internal.Isotope
            Either isotope name, ZAI, or internal representation of an isotope

        Returns
        -------
        float
            Atom density [atoms/b/cm]

        """
        return super().__getitem__(self._getIsotopeFromKey(key))

    def __repr__(self):
        return "<{} {} at {}>".format(self.__class__.__name__, self.name, hex(id(self)))

    def __str__(self):
        if self.mdens is not None:
            dens = " {:9.5e} [g/cc]".format(self.mdens)
        elif self.adens is not None:
            dens = " {:9.5e} [atoms/b/cm]".format(self.adens)
        else:
            dens = ""
        if self.temperature is not None:
            temp = " at {:7.5f} K".format(self.temperature)
        else:
            temp = ""
        if self.volume is not None:
            vol = " volume {:7.5f} cm^3".format(self.volume)
        else:
            vol = ""
        tail = "\n".join(
            "  {}: {:9.5e}".format(k.name, v) for k, v in sorted(self.items())
        )
        return "{cls} {name}{dens}{temp}{vol}\n{d}".format(
            cls=self.__class__.__name__,
            name=self.name,
            dens=dens,
            temp=temp,
            vol=vol,
            d=tail,
        )

    def copy(self, name=None):
        """Create a copy of this material

        Parameters
        ----------
        name : str, optional
            New name. Otherwise use :attr:`name`

        Returns
        -------
        hydep.Material

        """
        kwargs = {attr: getattr(self, attr)
                  for attr in ["adens", "mdens", "temperature", "volume"]}
        out = self.__class__(self.name if name is None else name, **kwargs)
        out.update(self)
        out.sab = self.sab.copy()
        return out

    def get(self, key: IsotopeLike, default: typing.Optional = None):
        """Retrieve the atom density for an isotope if it is contained

        Parameters
        ----------
        key : str or int or hydep.internal.Isotope
            Isotope name or ZAI, or the internal representation of an isotope
        default : object, optional
            Item to be returned if ``key`` does not correspond to an isotope
            on this material

        Returns
        -------
        object
            float of atom density [atoms/b/cm] if ``key`` matched an isotope
            stored on the material. Otherwise return ``default``

        Examples
        --------
        >>> w = Material("water", mdens=0.75)
        >>> w["H1"] = 0.047
        >>> w.get("H1")
        0.047
        >>> w.get("U235") is None
        True

        """
        iso = self._getIsotopeFromKey(key)
        return super().get(iso, default)

    def addSAlphaBeta(self, table: str):
        r"""Add S(alpha, beta) thermal scattering table

        Parameters
        ----------
        table : str
            Name of the :math:`S(\alpha,\beta)` thermal scattering
            table to be used in this material, e.g. ``"HinH20"``.
            If the :attr:`temperature` for this material is not set,
            it will be set to 600 K, as these tables are temperature
            dependent.

        Warns
        -----
        UserWarning
            If :attr:`temperature` is not set prior to calling this
            method.

        """
        if self.temperature is None:
            warnings.warn(
                f"Temperature on {self!r} not set. Defaulting to 600 K", UserWarning
            )
            self.temperature = 600
        self.sab.add(str(table))


class BurnableMaterial(Material):
    r"""Material to be burned comprised of isotopes and atom densities

    .. note::
        :attr:`volume` must be specified prior to running the
        simulation

    Parameters
    ----------
    name : str
        Name of this material
    adens : numbers.Real, optional
        Atomic density [atoms/b/cm] of this material. Mutually exclusive
        with ``mdens``
    mdens : numbers.Real, optional
        Mass density [g/cm^3] of this material. Mutually exclusive with
        ``adens``
    temperature : numbers.Real, optional
        Temperature [K] of this material
    volume : numbers.Real, optional
        Volume [cm^3] of this material
    **nucs :
        Additional keyword arguments are treated as nuclides
        and corresponding atomic densities.

    Attributes
    ----------
    name : str
        Name of this material
    adens : numbers.Real or None
        Atomic density [atoms/b/cm] of this material. Mutually exclusive
        with :attr:`mdens`
    mdens : numbers.Real or None
        Mass density [g/cm^3] of this material. Mutually exclusive with
        :attr:`adens`
    temperature : numbers.Real or None
        Temperature [K]
    volume : numbers.Real or None
        Volume [cm^3]
    sab : dict
        Not implemented, but will eventually support adding
        :math:`S(\alpha,\beta)` libraries for nuclides.
    id : int
        A unique postive identifier for this material out of all
        :class:`hydep.Material`
    index : Optional[int]
        Non-negative index for this material out of all
        :class:`hydep.BurnableMaterial` instances in a given
        :class:`hydep.Model`. Only allowed to be set once.

    """

    def __init__(
        self, name, adens=None, mdens=None, temperature=None, volume=None, **nucs
    ):
        super().__init__(
            name,
            adens=adens,
            mdens=mdens,
            temperature=temperature,
            volume=volume,
            **nucs,
        )
        self.microxs = None
        self._index = None

    def asVector(self, order=None, default=0.0):
        """Return a vector of atom densities given some ordering

        Parameters
        ----------
        order : Dict[int, int] or Iterable[int], optional
            Order to write isotopes. If not provided, write all
            isotopes sorted by their :term:`ZAI` number. If ``order``
            is a map, it should map :term:`ZAI`s to indexes in the
            resulting array. Otherwise it should be an iterable, e.g.
            :class:`list` or :class:`numpy.ndarray` of :term:`ZAI`.
            Isotopes will be written in this order
        default : float, optional
            Default value to write if an isotope is present in ``order``
            but not on this material.

        Returns
        -------
        numpy.ndarray
            Atom densities for this material according to the ordering.
            Length is equal to the number of isotopes in the material,
            or in ordering if provided.

        """
        if order is None:
            return numpy.fromiter((self[k] for k in sorted(self)), dtype=float)
        elif isinstance(order, Mapping):  # zai -> index
            out = numpy.empty(len(order))
            for z, ix in order.items():
                out[ix] = self.get(z, default)
                return out
        elif isinstance(order, Iterable):  # zai
            out = numpy.empty(len(order))
            for ix, z in enumerate(order):
                out[ix] = self.get(z, default)
            return out
        else:
            raise TypeError("Ordering {} not understood".format(order))

    def __array__(self):
        """Convert directly to numpy array using dispatching"""
        return self.asVector(order=None)  # VER numpy >= 1.16

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        if self._index is not None:
            raise AttributeError("Index already set for {}".format(self))

        if not isinstance(value, numbers.Integral):
            value = int(value)

        if value < 0:
            raise ValueError("Index cannot be negative")

        self._index = value
