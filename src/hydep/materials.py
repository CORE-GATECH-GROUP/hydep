import numbers
from collections.abc import Mapping, Iterable

import numpy

from hydep.typed import TypedAttr, BoundedTyped
from hydep.internal import Isotope, ZaiTuple, getIsotope, TemporalMicroXs


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
    sab : dict
        Not implemented, but will eventually support adding
        :math:`S(\alpha,\beta)` libraries for nuclides.
    id : str
        A unique identifier for this material.

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
        self.sab = {}  # TODO this
        super().__init__()
        self.update(nucs)

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

    def __setitem__(self, key, value):
        assert isinstance(value, numbers.Real)
        assert value > 0
        super().__setitem__(self._getIsotopeFromKey(key), value)

    def __getitem__(self, key):
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

    @property
    def id(self):
        return hex(id(self))

    def copy(self, name=None):
        kwargs = {attr: getattr(self, attr)
                  for attr in ["adens", "mdens", "temperature", "volume"]}
        out = self.__class__(self.name if name is None else name, **kwargs)
        out.update(self)
        return out


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
    id : str
        A unique identifier for this material
    microxs : hydep.internal.TemporalMicroXs or None
        Container for microscopic cross sections over time

    """

    microxs = TypedAttr("microxs", TemporalMicroXs, allowNone=True)
    counts = TypedAttr("counts", int)

    def __init__(
        self, name, adens=None, mdens=None, temperature=None, volume=None, **nucs
    ):
        super().__init__(
            name,
            adens=adens,
            mdens=mdens,
            temperature=temperature,
            volume=volume,
            **nucs
        )
        self.microxs = None
        self.counts = 0

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
