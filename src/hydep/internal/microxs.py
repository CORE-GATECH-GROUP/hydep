"""
Class for handling vectorized microscopic cross sections
"""
import bisect
import collections
import numbers
import typing

import numpy
from numpy.polynomial.polynomial import polyfit, polyval

from .timetravel import CachedTimeTraveller

# TODO Is the sparse structure worth it? Most isotopes have a few reactions
# TODO of interest, with some having no more than 10


class MicroXsVector:
    """Microscopic cross sections for a single material

    Intended to be used in conjunction with :class:`TemporalMicroXs`
    to help with extrapolation, and in depletion. Mimics a sparse
    array, where :attr:`zai` has fewer elements, but iteration recovers
    the full ``(zai, reaction mt, xs)`` triplet. For consistent
    construction, it is recommended to use :meth:`fromLongFormVectors`.

    Parameters
    ----------
    zai : iterable of int
        Sorted vector containing the ZAI identifiers for reactions in
        this instance. Expected to be sorted in increasing order,
        and with length less than or equal to ``rxns``
    zptr: iterable of int
        Pointer vector that describes where reactions for each isotope
        are found
    reactions : iterable of int
        Reaction MTS (e.g. 102, 18) such that reactions
        ``rxns[zptr[i]:zptr[i+1]]`` belong to isotope ``zai[i]``
    mxs : numpy.ndarray
        Microscopic cross sections. Of similar size and ordering to
        ``reactions, such ``mxs[zptr[i]:zptr[i+1]]`` are of MT
        ``rxns[zptr[i]:zptr[i+1]]`` and belong to isotope ``zai[i]``

    Attributes
    ----------
    zai : tuple of int
        Sorted vector containing the ZAI identifiers for reactions in
        this instance. Expected to be sorted in increasing order,
        and with length less than or equal to ``rxns``
    zptr: tuple of int
        Pointer vector that describes where reactions for each isotope
        are found
    reactions : tuple of int
        Reaction MTS (e.g. 102, 18) such that reactions
        ``rxns[zptr[i]:zptr[i+1]]`` belong to isotope ``zai[i]``
    mxs : numpy.ndarray
        Microscopic cross sections. Of similar size and ordering to
        ``reactions, such ``mxs[zptr[i]:zptr[i+1]]`` are of MT
        ``rxns[zptr[i]:zptr[i+1]]`` and belong to isotope ``zai[i]``

    Examples
    --------
    >>> zai = [922350, 922350, 541350, 922380]
    >>> rxns = [102, 18, 102, 18]
    >>> values = [0.4, 0.05, 10, 0.01]
    >>> microxs = MicroXsVector.fromLongFormVectors(
    ...     zai, rxns, values, assumeSorted=False)
    >>> microxs.zai
    (541350, 922350, 922380)
    >>> microxs.zptr
    (0, 1, 3, 4)
    >>> microxs.rxns == (102, 102, 18, 18)
    True
    >>> all(microxs.mxs == (10, 0.4, 0.05, 0.01))
    True
    >>> new = microxs * 2
    >>> all(new.mxs == (20, 0.8, 0.1, 0.02))
    True
    >>> new *= 0.5
    >>> all(new.mxs == [10, 0.4, 0.05, 0.01])
    True

    """

    __slots__ = ("zai", "zptr", "rxns", "mxs")

    def __init__(self, zai, zptr, rxns, mxs):
        self.zai = tuple(zai)
        self.zptr = tuple(zptr)
        self.rxns = tuple(rxns)
        self.mxs = mxs

    @classmethod
    def fromLongFormVectors(cls, zaiVec, rxnVec, mxs, assumeSorted=True):
        """Construction with equal length, potentially unsorted vectors

        Parameters
        ----------
        zaiVec : Sequence[int]
            Vector of isotope ZAI numbers
        rxnVec : Sequence[int]
            Vector of reaction MT numbers
        mxs : Sequence[float]
            Vector of microscopic cross sections such that reaction
            ``rxnVec[i] == r``, maybe 18, of isotope
            ``zaiVec[i] == z``, maybe ``922350``, has a value of
            ``mxs[i]``
        assumeSorted : bool
            Skip sorting by ZAI under the assumption that the vectors
            are sorted accordingly

        Returns
        -------
        MicroXsVector

        """

        zaiVec = numpy.asarray(zaiVec, dtype=int).copy()
        if not isinstance(zaiVec[0], numbers.Integral):
            raise TypeError(
                "Vectors must be vectors of integer, zaiVec[0] "
                "became {}".format(zaiVec[0])
            )

        rxnVec = numpy.asarray(rxnVec, dtype=int).copy()
        mxs = numpy.asarray(mxs).copy()

        if zaiVec.shape != rxnVec.shape != mxs.shape:
            raise ValueError(
                "Shapes are inconsistent. ZAI: {}, rxn: {}, mxs: {}".format(
                    zaiVec.shape, rxnVec.shape, mxs.shape
                )
            )

        if not assumeSorted:
            sortIx = numpy.argsort(zaiVec)
            zaiVec = zaiVec[sortIx]
            rxnVec = rxnVec[sortIx]
            mxs = mxs[sortIx]

        zai = [zaiVec[0]]
        zptr = [0]
        for ix, z in enumerate(zaiVec):
            if z == zai[-1]:
                continue
            zai.append(z)
            zptr.append(ix)

        zptr.append(rxnVec.size)

        return cls(zai, zptr, rxnVec, mxs)

    def __iter__(self):
        start = self.zptr[0]
        for ix, z in enumerate(self.zai):
            stop = self.zptr[ix + 1]
            for rxn, mxs in zip(self.rxns[start:stop], self.mxs[start:stop]):
                # TODO namedtuple?
                yield z, rxn, mxs
            start = stop

    def __len__(self):
        return len(self.rxns)

    def __imul__(self, scalar):
        if not isinstance(scalar, numbers.Real):
            return NotImplemented
        self.mxs *= scalar
        return self

    def __mul__(self, scalar):
        if not isinstance(scalar, numbers.Real):
            return NotImplemented
        new = self.__class__(self.zai, self.zptr, self.rxns, self.mxs.copy())
        new *= scalar
        return new

    def __rmul__(self, scalar):
        return self * scalar

    def getReactions(self, zai: numbers.Integral, default=None):
        """Retrieve all reactions for a given isotope, if present

        Parameters
        ----------
        zai : int
            Isotope ZAI identifier, e.g. ``922350``
        default : optional
            Item to be returned if ``zai`` is not found. Defaults
            to ``None``

        Returns
        -------
        dict or type(default)
            If ``zai`` is found, return a dictionary of ``{rxn: xs}``,
            where ``rxn`` is an integer reaction number and ``xs`` is
            the, potentially multi-group, cross section for
            ``zai, rxn``. If ``zai`` is not found, return ``default``

        """
        assert isinstance(zai, numbers.Integral)
        ix = bisect.bisect_left(self.zai, zai)
        if ix == len(self.zai) or self.zai[ix] != zai:
            return default
        s = slice(self.zptr[ix], self.zptr[ix + 1])
        return dict(zip(self.rxns[s], self.mxs[s]))

    def getReaction(self, zai, rxn, default=None):
        """Retrieve a specific reaction, if present

        Parameters
        ----------
        zai : int
            Isotope ZAI identifier, e.g. ``922350``
        rxn : int
            Reaction MT of interest
        default : optional
            Item to be returned if reaction ``rxn`` for isotope
            ``zai`` is not found. Defaults to None

        Returns
        -------
        float or iterable of float or type(default)
            Potentially multi-group cross section for reaction ``rxn``
            of isotope ``zai`` if found. Otherwise return ``default``

        """
        assert isinstance(zai, numbers.Integral)
        ix = bisect.bisect_left(self.zai, zai)
        if ix == len(self.zai) or self.zai[ix] != zai:
            return default
        s = slice(self.zptr[ix], self.zptr[ix + 1])
        for r, m in zip(self.rxns[s], self.mxs[s]):
            if r == rxn:
                return m
        else:
            return default


ArrayOrMicroXsVector = typing.Union[numpy.ndarray, MicroXsVector]


class TemporalMicroXs(CachedTimeTraveller):
    """Microscopic cross sections over time

    Uses a similar sparse structure as :class:`MicroXsVector`.

    Parameters
    ----------
    zai : iterable of int
        Isotope ZAI identifiers. Will have a length less than
        or equal to ``rxns``
    zptr : iterable of int
        Pointer vector indicating that reactions for isotope
        ``zai[i]`` can be found in ``rxns[zptr[i:i+1]``
    rxns : iterable of int
        Tuple of reaction numbers (MTs)
    mxs : iterable of MicroXsVector, optional
        Initial microscopic cross section vectors to be loaded.
        Must be consistent with ``zai``, ``zptr``, and ``rxns``
        parameters passed. Assumed to correspond to each point in
        ``time``.
    time : iterable of float, optional
        Initial time points for each value in ``mxs``. Can be sorted
        or not, but ``assumeSorted`` should be passed accordingly
    maxlen : int, optional
        Number of time points and microscopic cross section vectors to
        be stored at any one instance.
    order : int, optional
        Fitting order, e.g. assume constant cross sections if ``order==0``,
        linear for ``order==1``, quadratic for ``order==2``, etc.
    assumeSorted : bool, optional
        If initial time and microsopic cross sections are provided, but
        not sorted, pass a true value. This will insert the values in a
        sorted manner.

    Attributes
    ----------
    zai : tuple of int
        Isotope ZAI identifiers
    zptr : tuple of int
        Pointer vector detailing where reactions and cross sections are located
        for specific isotopes
    rxns : tuple of int
        Reaction numbers
    order : int
        Fitting order
    time : tuple of float
        Copy of time points stored on the object

    See Also
    --------
    * :meth:`fromMicroXsVector` - helper construction method

    """

    # TODO Hide / control access to attributes that shouldn't be touched

    __slots__ = CachedTimeTraveller.__slots__ + (
        "zai",
        "zptr",
        "rxns",
        "_mxs",
        "order",
        "_coeffs",
    )

    def __init__(
        self, zai, zptr, rxns, mxs=None, time=None, maxlen=3, order=1, assumeSorted=True
    ):
        self.zai = tuple(zai)
        self.zptr = tuple(zptr)
        self.rxns = tuple(rxns)

        if (time is not None) != (mxs is not None):
            raise ValueError("time and mxs must both be None or provided")

        if maxlen is not None:
            assert order < maxlen, (order, maxlen)
        self.order = order

        super().__init__(maxlen)
        self._mxs = collections.deque(maxlen=maxlen)

        if time is not None:
            self.extend(time, mxs)

        self._coeffs = None

    @classmethod
    def fromMicroXsVector(cls, mxsVector, time, maxlen=3, order=1):
        """Construct given a single set of cross sections

        Parameters
        ----------
        mxsVector : MicroXsVector
            Set of cross sections to be treated as starting data
        time : float
            Point in time at which these cross section were generated
        maxlen : Optional[int]
            Maximum number of points to create space for on the
            outgoing object. Defaults to three
        order : Optional[int]
            Fitting order

        Returns
        -------
        TemporalMicroXs

        """
        return cls(
            mxsVector.zai,
            mxsVector.zptr,
            mxsVector.rxns,
            mxs=(mxsVector.mxs,),
            time=(time,),
            maxlen=maxlen,
            order=order,
        )

    def _check(self, xsvector):
        if isinstance(xsvector, MicroXsVector):
            if (
                xsvector.zai != self.zai
                or xsvector.zptr != self.zptr
                or xsvector.rxns != self.rxns
            ):
                raise ValueError(
                    "Incoming cross sections do not align with currently stored "
                    "values. In the future, passing subsets may be supported."
                )
            mxs = xsvector.mxs
        else:
            mxs = numpy.asarray(xsvector)

        if len(mxs) != len(self.rxns):
            raise ValueError(
                "Expected array with {} reactions, got {}".format(
                    len(self.rxns), len(mxs)
                )
            )
        if self and mxs.shape != self._mxs[0].shape:
            raise ValueError(
                "Expected array with {} reactions and {} groups, got {}".format(
                    *self._mxs[0].shape, mxs.shape
                )
            )
        return mxs

    def __getitem__(self, index: int) -> MicroXsVector:
        return MicroXsVector(self.zai, self.zptr, self.rxns, self._mxs[index])

    def __setitem__(self, index: int, mxs: ArrayOrMicroXsVector):
        """Overwrite the cross sections at index ``index``

        Parameters
        ----------
        index : int
            Index of the time slot that should be overwritten
        mxs : MicroXsVector or numpy.ndarray
            If a cross section vector, must have identical
            :attr:`zai`, :attr:`zptr`, and :attr:`rxns` data.
            Number of energy groups in :attr:`MicroXsVector.mxs`
            must be consistent as well. Otherwise, must reflect
            some 2D array of cross sections, ``(reactions, group)``
            that is consistent with what is currently stored.

        """
        valid = self._check(mxs)
        self._mxs[index] = valid

    def insort(self, time: float, mxs: ArrayOrMicroXsVector):
        """Insert microscopic cross sections to maintain ordering in time

        Parameters
        ----------
        time : float
            Point in calendar time [s] that generated the cross sections
        mxs : MicroXsVector or numpy.ndarray
            If a cross section vector, must have identical
            :attr:`zai`, :attr:`zptr`, and :attr:`rxns` data.
            Number of energy groups in :attr:`MicroXsVector.mxs`
            must be consistent as well. Otherwise, must reflect
            some 2D array of cross sections, ``(reactions, group)``
            that is consistent with what is currently stored.

        """
        value = self._check(mxs)
        return super().insort(time, value)

    def _insert(self, index: int, mxs: numpy.ndarray) -> None:
        self._mxs.insert(index, mxs)
        self._coeffs = None

    def append(self, time: float, mxs: ArrayOrMicroXsVector):
        """Add cross sections from a later point in time

        Time must be greater than the greatest value currently
        stored. Otherwise use :meth:`insert`.

        Parameters
        ----------
        time : float
            Point in calendar time [s] that generated the cross sections
        mxs : MicroXsVector or numpy.ndarray
            If a cross section vector, must have identical
            :attr:`zai`, :attr:`zptr`, and :attr:`rxns` data.
            Number of energy groups in :attr:`MicroXsVector.mxs`
            must be consistent as well. Otherwise, must reflect
            some 2D array of cross sections, ``(reactions, group)``
            that is consistent with what is currently stored.

        """
        value = self._check(mxs)
        return super().append(time, value)

    def _append(self, mxs: numpy.ndarray):
        self._mxs.append(mxs)
        self._coeffs = None

    def extend(
        self, times: typing.Sequence[float], mxs: typing.Sequence[ArrayOrMicroXsVector]
    ):
        """Add a collection times and cross sections to the deque

        Parameters
        ----------
        times : Sequence of float
            Point in calendar time [s] that generated the cross sections
        mxs : Sequence of numpy.ndarray or sequence of MicroXsVector
            If cross section vectors, must have identical
            :attr:`zai`, :attr:`zptr`, and :attr:`rxns` data.
            Number of energy groups in :attr:`MicroXsVector.mxs`
            must be consistent as well. Otherwise, must reflect
            some 2D array of cross sections, ``(reactions, group)``
            that is consistent with what is currently stored.
        """
        return super().extend(times, mxs)

    def _extend(self, values) -> None:
        extensions = []
        for m in values:
            extensions.append(self._check(m))
        self._mxs.extend(extensions)
        self._coeffs = None

    def __call__(self, time: float) -> MicroXsVector:
        """Evaluate the microscopic cross sections at a given time

        Parameters
        ----------
        time : float
            Point in time, consistent with units on :attr:`time`,
            to perform the fitting

        Returns
        -------
        MicroXsVector
            Microscopic cross sections evaluated at ``time``

        Raises
        ------
        ValueError
            If no cross sections have been added for fitting

        """
        if not self:
            raise ValueError("No cross sections added for evaluation")
        return super().__call__(time)

    def _evaluate(self, time: float) -> MicroXsVector:
        if self._coeffs is None:
            self._coeffs = self._genCoeffs()
        mxs = numpy.empty(self._mxs[0].shape)
        for ix, c in enumerate(self._coeffs):
            mxs[ix] = polyval(time, c)
        return MicroXsVector(self.zai, self.zptr, self.rxns, mxs)

    def _genCoeffs(self):
        # convert (time, reaction, group) -> (reaction, time, group)
        data = numpy.array(self._mxs).transpose(1, 0, 2)
        coeffs = numpy.empty((data.shape[0], self.order + 1, data.shape[2]))
        # TODO Vectorize?
        for index, rxn in enumerate(data):
            # group, time
            coeffs[index] = polyfit(self._times, rxn, self.order, full=False)

        return coeffs
