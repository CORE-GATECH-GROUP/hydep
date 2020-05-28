"""
Classes for interacting with microscopic cross sections and reaction rates

"""
import math
from collections import deque
from collections.abc import Iterable
import typing
import bisect
import numbers

import numpy
from numpy.polynomial import polynomial


class XsIndex:
    """Index and locator for microscopic cross sections and reaction rates

    The incoming parameters and their corresponding attributes :attr:`zais`,
    :attr:`rxns`, and :attr:`zptr` are expected to be arranged such that
    ``rxns[zptr[i]:zptr[i+1]`` correspond to isotope ``zais[i]``.

    Parameters
    ----------
    zais : iterable of int
        Sorted isotope ZAI identifiers, must be in increasing ordering
    rxns : iterable of int
        Reactions for each isotope in ``zais``. Expected to be of
        length ``>= len(zais)``
    zptr : iterable of int
        Pointer vector describing which reactions correspond to which
        isotopes. Must have length one greater than ``zais`` where
        ``zptr[-1] == len(rxns)``

    Attributes
    ----------
    zais : tuple of int
        Isotope ZAI identifiers
    rxns : tuple of int
        Reaction MT numbers
    zptr : tuple of int
        Pointer vector

    Examples
    --------

    >>> xs = XsIndex(
    ...     [80160, 922350, 922380],
    ...     [102, 18, 102, 102, 18],
    ...     [0, 1, 3, 5],
    ... )
    >>> list(xs)
    [(80160, 102), (922350, 18), (922350, 102), (922380, 102), (922380, 18)]
    >>> xs[2]
    (922350, 102)
    >>> xs(922350, 102)
    2
    >>> xs[-1]
    (922380, 18)
    >>> xs.findZai(922380)
    2
    >>> for rxn, index in xs.getReactions(922350):
    ...     print(rxn, index)
    18 1
    102 2

    """

    def __init__(
        self,
        zais: typing.Iterable[int],
        rxns: typing.Iterable[int],
        zptr: typing.Iterable[int],
    ):
        if len(zais) != len(zptr) - 1:
            raise ValueError(
                f"Expect one additional pointer than isotopes. Got {len(zptr)} "
                f"pointers and {len(zais)} isotopes"
            )
        if len(rxns) != zptr[-1]:
            raise ValueError(
                f"Number of reactions {len(rxns)} not equal to final pointer "
                f"{zptr[-1]}"
            )
        self._zais = tuple(zais)
        self._zptr = tuple(zptr)
        self._rxns = tuple(rxns)

    def __len__(self) -> int:
        """Number of stored reactions"""
        return len(self._rxns)

    def __iter__(self) -> typing.Iterator[typing.Tuple[int, int]]:
        """Iterate over ``(zai, rxn)`` pairs"""
        for ix, z in enumerate(self._zais):
            for rxn in self._rxns[self._zptr[ix] : self._zptr[ix + 1]]:
                yield z, rxn

    def __eq__(self, other: "XsIndex") -> bool:
        """Return True of ``other`` has same isotope and reaction data"""
        if isinstance(other, type(self)):
            return (
                self._zais == other.zais
                and self._zptr == other.zptr
                and self._rxns == other.rxns
            )
        return NotImplemented

    def __hash__(self) -> int:
        """Hash of indexing information"""
        return hash(self._zais + self._zptr + self._rxns)

    def __getitem__(self, ix) -> typing.Tuple[int, int]:
        """Retrieve the isotope and reaction at a given index

        Parameters
        ----------
        ix : int
            Index in :attr:`rxns` of interest

        Returns
        -------
        int
            Isotope ZAI for this reaction
        int
            Reaction number

        """
        if ix < 0:
            ix = len(self._rxns) + ix
        rxn = self._rxns[ix]
        place = bisect.bisect_left(self._zptr, ix)
        if self._zptr[place] == ix:
            return self._zais[place], rxn
        return self._zais[place - 1], rxn

    def __call__(self, zai: int, rxn: int) -> int:
        """Retrieve the index for a given isotope and reaction

        Parameters
        ----------
        zai : int
            Isotope ZAI
        rxn : int
            Reaction MT

        Returns
        -------
        int
            Index for this reaction. Symmetric with
            :meth:`__getitem__`

        Raises
        ------
        ValueError
            If ``zai`` or ``rxn`` was not found

        """
        for myrxn, ix in self.getReactions(zai):
            if myrxn == rxn:
                return ix
        raise ValueError(f"Reaction {rxn} of isotope {zai} not found")

    @property
    def zais(self):
        return self._zais

    @property
    def rxns(self):
        return self._rxns

    @property
    def zptr(self):
        return self._zptr

    def findZai(self, zai: int) -> int:
        """Return the index in :attr:`zais` for a given ZAI

        Parameters
        ----------
        zai : int
            Isotope ZAI of interest

        Returns
        -------
        int
            Index in :attr:`zais` such that
            ``x.zais[x.findZai(z)] == z``

        Raises
        ------
        ValueError
            If ``zai`` is not found

        """
        zix = bisect.bisect_left(self._zais, zai)
        if zix == len(self._zais) or self._zais[zix] != zai:
            raise ValueError(f"{zai} not found")
        return zix

    def getReactions(
        self, zai: int
    ) -> typing.Generator[typing.Tuple[int, int], None, None]:
        """Return indexes for all reactions of a given isotope

        Parameters
        ----------
        zai : int
            Isotope ZAI of interest

        Returns
        -------
        generator of (int, int)
            Object that yields reaction mt and corresponding index
            for all reactions of isotope ``zai``

        Raises
        ------
        ValueError
            If ``zai`` is not found

        """
        zix = self.findZai(zai)
        start, end = self._zptr[zix : zix + 2]
        return (
            (rxn, start + ix) for ix, rxn in enumerate(self.rxns[start:end])
        )


class _IndexedData:
    def __init__(self, reactionIndex: XsIndex, data: numpy.ndarray):
        self.index = reactionIndex
        self.data = data


class MaterialData(_IndexedData):
    """Cross section or reaction rates for a single material

    Mainly an intermediate class, dispatched by :class:`MaterialDataArray`
    when iterating or accessing specific elements

    Parameters
    ----------
    reactionIndex : XsIndex
        Index to describe the ordering of isotopes and reactions
    data : numpy.ndarray
        1-D vector describing cross sections or reaction rates for
        each ``(zai, rxn)`` pair in ``reactionIndex``

    Attributes
    ----------
    reactionIndex : XsIndex
        Index to describe the ordering of isotopes and reactions
    data : numpy.ndarray
        1-D vector describing cross sections or reaction rates for
        each ``(zai, rxn)`` pair in :attr:`reactionIndex`

    """

    def getReactions(
        self, zai: int, default: typing.Optional[typing.Any] = None
    ):
        """Obtain data for all reactions for a specific isotope

        Parameters
        ----------
        zai : int
            Isotope ZAI of interest
        default : object, optional
            Object to return if ``zai`` is not found

        Returns
        -------
        object
            If ``zai`` is found, will be a dictionary mapping
            ``{int: float}`` where each ``int`` is a reaction MT
            mapped to the corresponding cross sections or reaction
            rates

        """
        try:
            reactionIndices = self.index.getReactions(zai)
        except ValueError:
            return default
        return {rxn: self.data[ix] for rxn, ix in reactionIndices}


class MaterialDataArray(_IndexedData):
    """Collection of data for several materials

    Parameters
    ----------
    reactionIndex : XsIndex
        Index to describe the ordering of isotopes and reactions
    data : numpy.ndarray
        2D array of shape ``(N_mats, N_rxns)`` such that
        ``data[i, j]`` corresponds to the reaction for
        ``zai, rxnMT = reactionIndex[j]`` for material ``i``

    Attributes
    ----------
    reactionIndex : XsIndex
        Index to describe the ordering of isotopes and reactions
    data : numpy.ndarray
        2D array of shape ``(N_mats, N_rxns)`` such that
        ``data[i, j]`` corresponds to the reaction for
        ``zai, rxnMT = reactionIndex[j]`` for material ``i``

    """

    def __getitem__(self, pos: int) -> MaterialData:
        """Obtain a view into material data for a given material index"""
        if isinstance(pos, numbers.Integral):
            return MaterialData(self.index, self.data[pos])
        raise TypeError(
            f"Only integer access allowed for {self}, not {type(pos)}"
        )

    def __iter__(self) -> typing.Iterator[MaterialData]:
        """Iterate over data for all materials on the object"""
        return (MaterialData(self.index, row) for row in self.data)

    def __len__(self) -> int:
        """Number of materials stored"""
        return len(self.data)


class DataBank:
    """Store and extrapolate :class:`MaterialDataArray`s at unique time points

    Parameters
    ----------
    nsteps : int
        Number of time points and material arrays to store
    nmaterials : int
        Number of materials that will be stored at each time point
    rxnIndex : XsIndex
        Indexer that describes the ordering of cross section data
    order : int, optional
        Maximum polynomial fitting order. Default is one (linear)

    """

    def __init__(
        self,
        nsteps: int,
        nmaterials: int,
        rxnIndex: XsIndex,
        order: typing.Optional[int] = 1,
    ):
        self._data = numpy.empty(
            (nsteps, nmaterials, len(rxnIndex)), dtype=numpy.float64
        )
        self._reactionIndex = rxnIndex
        self._times = numpy.empty(nsteps)
        self._timeIndex = deque(maxlen=nsteps)
        self._coeffs = None
        self._order = order

    @property
    def reactionIndex(self) -> XsIndex:
        """Read-only property for underlying reaction index"""
        return self._reactionIndex

    @property
    def shape(self) -> typing.Tuple[int, int, int]:
        """Shape of underlying array"""
        return self._data.shape

    @property
    def nsteps(self) -> int:
        """Number of time steps that could be stored"""
        return self._data.shape[0]

    @property
    def nmaterials(self) -> int:
        """Number of materials that are stored for each time point"""
        return self._data.shape[1]

    @property
    def nreactions(self) -> int:
        """Number of reactions stored"""
        return self._data.shape[2]

    @property
    def stacklen(self) -> int:
        """Number of time points currently stored"""
        return len(self._timeIndex)

    def push(self, t: float, materialData: MaterialDataArray):
        """Push another set of data to be extrapolated

        Time value are expected to be pushed in order of
        increasing time.

        Parameters
        ----------
        t : float
            Point in calendar time at which ``materialData`` was
            generated. Units on time should be consistent with
            subsequent calls to :meth:`at` and :meth:`getReactionRatesAt`
        materialData : MaterialDataArray
            Cross section data for all materials at this point in time.
            Must have a consistent index with :attr:`reactionIndex`

        """
        if materialData.index != self._reactionIndex:
            raise ValueError("Reaction indices do not conform")
        if self._timeIndex:
            if t <= self._times[self._timeIndex[-1]]:
                raise ValueError(
                    f"Current time {t} is less than maximum time "
                    f"{self._times[self._timeIndex[-1]]}"
                )
            index = (self._timeIndex[-1] + 1) % self._data.shape[0]
        else:
            index = 0
        self._data[index] = materialData.data
        self._timeIndex.append(index)
        self._times[index] = t
        self._coeffs = None

    def at(
        self, t: float, atol: typing.Optional[float] = 1e-12
    ) -> MaterialDataArray:
        """Project cross sections to a new point in time

        Will make a polynomial fitting based on the number
        of current points stored and the supplied fitting
        order. If the order exceeds the maximum polynomial
        order that could be built, this maximum order will be
        used, e.g. will not use a quadratic fit until three or
        more points have been loaded.

        Parameters
        ----------
        t : float
            Point in calendar time at which cross sections are
            requested. Units should be consistent with units
            used in previous calls to :meth:`push`
        atol : float, optional
            Absolute tolerance to use when checking if ``t``
            corresponds to a currently stored point in
            time

        Returns
        -------
        MaterialDataArray
            Cross sections at the requested point in time

        Raises
        ------
        AttributeError
            If no calls to :meth:`push` have been made and
            there is no data to be projected

        """
        if not self._timeIndex:
            raise AttributeError("No xs loaded")

        # Sequential search because we should only be storing
        # a few time points, and time vector is not sorted
        for ix in self._timeIndex:
            if math.fabs(self._times[ix] - t) <= atol:
                vals = self._data[ix]
                break
        else:
            if self._coeffs is None:
                self._coeffs = polynomial.polyfit(
                    self._times[self._timeIndex],
                    self._data[self._timeIndex].reshape(
                        len(self._timeIndex), self.nmaterials * self.nreactions
                    ),
                    deg=min(len(self._timeIndex) - 1, self._order),
                )

            vals = polynomial.polyval(t, self._coeffs).reshape(
                self.nmaterials, self.nreactions
            )

        return MaterialDataArray(self._reactionIndex, vals)

    def getReactionRatesAt(
        self,
        t: float,
        fluxes: numpy.ndarray,
        atol: typing.Optional[float] = 1e-12,
    ) -> MaterialDataArray:
        """Project cross sections and then compute reaction rates

        Parameters
        ----------
        t : float
            Point in calendar time at which reaction rates are
            requested. Units should be consistent with units
            from :attr:`push`
        fluxes : iterable of float
            1-D vector of one-group scalar flux [n/cm2/s] in each
            burnable material
        atol : float, optional
            Absolute tolerance used when determining if ``t``
            corresponds to a previously computed point

        Returns
        -------
        MaterialDataArray
            Reaction rates in each material with an index to determine
            how the reactions are ordered

        """
        if not isinstance(fluxes, Iterable):
            raise TypeError(
                f"Fluxes must be vector of float, not {type(fluxes)}"
            )
        elif len(fluxes) != self.nmaterials:
            raise ValueError(
                f"Was given {len(fluxes)} fluxes for {self.nmaterials} "
                "burnable materials"
            )
        try:
            upcast = numpy.broadcast_to(
                fluxes, (self.nmaterials, len(self._reactionIndex)),
            )
        except ValueError as ve:
            raise ValueError(
                f"Failed to coerce fluxes of shape {numpy.shape(fluxes)} to shape "
                f"({self.nmaterials}, {len(self._reactionIndex)})"
            ) from ve

        mxs = self.at(t, atol=atol)
        mxs.data = mxs.data * upcast
        return mxs
