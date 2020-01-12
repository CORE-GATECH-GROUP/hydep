"""
Class for handling vectorized microscopic cross sections
"""
import bisect
import collections
import numbers

import numpy
from numpy.polynomial.polynomial import polyfit, polyval


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
    zai : numpy.ndarray of int
        Sorted vector containing the ZAI identifiers for reactions in
        this instance. Expected to be sorted in increasing order,
        and with length less than or equal to ``rxns``
    zptr: numpy.ndarray of int
        Pointer vector that describes where reactions for each isotope
        are found
    reactions : numpy.ndarray of int
        Reaction MTS (e.g. 102, 18) such that reactions
        ``rxns[zptr[i]:zptr[i+1]]`` belong to isotope ``zai[i]``
    mxs : numpy.ndarray
        Microscopic cross sections. Of similar size and ordering to
        ``reactions, such ``mxs[zptr[i]:zptr[i+1]]`` are of MT
        ``rxns[zptr[i]:zptr[i+1]]`` and belong to isotope ``zai[i]``

    Attributes
    ----------
    zai : numpy.ndarray of int
        Sorted vector containing the ZAI identifiers for reactions in
        this instance. Expected to be sorted in increasing order,
        and with length less than or equal to ``rxns``
    zptr: numpy.ndarray of int
        Pointer vector that describes where reactions for each isotope
        are found
    reactions : numpy.ndarray of int
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
    array([541350, 922350, 922380])
    >>> microxs.zptr
    array([0, 1, 3, 4])
    >>> all(microxs.rxns == [102, 102, 18, 18])
    True
    >>> all(microxs.mxs == [10, 0.4, 0.05, 0.01])
    True
    >>> new = microxs * 2
    >>> all(new.mxs == [20, 0.8, 0.1, 0.02])
    True
    >>> new *= 0.5
    >>> all(new.mxs == [10, 0.4, 0.05, 0.01])
    True

    """

    __slots__ = ("zai", "zptr", "rxns", "mxs")

    def __init__(self, zai, zptr, rxns, mxs):
        self.zai = zai
        self.zptr = zptr
        self.rxns = rxns
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
                "became {}".format(zaiVec[0]))

        rxnVec = numpy.asarray(rxnVec, dtype=int).copy()
        mxs = numpy.asarray(mxs).copy()

        if zaiVec.shape != rxnVec.shape != mxs.shape:
            raise ValueError(
                "Shapes are inconsistent. ZAI: {}, rxn: {}, mxs: {}".format(
                    zaiVec.shape, rxnVec.shape, mxs.shape))

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

        return cls(numpy.array(zai), numpy.asarray(zptr), rxnVec, mxs)

    def __iter__(self):
        start = self.zptr[0]
        for ix, z in enumerate(self.zai):
            stop = self.zptr[ix + 1]
            for rxn, mxs in zip(self.rxns[start:stop], self.mxs[start:stop]):
                # TODO namedtuple?
                yield z, rxn, mxs
            start = stop

    def __len__(self):
        return self.rxns.shape[0]

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


class TemporalMicroXs:
    """Microscopic cross sections over time

    Uses a similar sparse structure as :class:`MicroXsVector`.

    Parameters
    ----------
    zai : Tuple[int...]
        Isotope ZAI identifiers. Will have a length less than
        or equal to ``rxns``
    zptr : Tuple[int...]
        Pointer vector indicating that reactions for isotope
        ``zai[i]`` can be found in ``rxns[zptr[i:i+1]``
    rxns : Tuple[int...]
        Tuple of reaction numbers (MTs)
    mxs : Optional[Iterable[MicroXsVector]]
        Initial microscopic cross section vectors to be loaded.
        Assumed to correspond to each point in ``time``.
    time : Optional[Iterable[float]]
        Initial time points for each value in ``mxs``. Can be sorted
        or not, but ``assumeSorted`` should be passed accordingly
    maxlen : Optional[int]
        Number of time points and microscopic cross section vectors to
        be stored at any one instance.
    order : Optional[int]
        Fitting order, e.g. assume constant cross sections if ``order==0``,
        linear for ``order==1``, quadratic for ``order==2``, etc.
    assumeSorted : Optional[bool]
        If initial time and microsopic cross sections are provided, but
        not sorted, pass a true value. This will insert the values in a
        sorted manner.

    Attributes
    ----------
    zai : Tuple[int...]
        Isotope ZAI identifiers
    zptr : Tuple[int...]
        Pointer vector detailing where reactions and cross sections are located
        for specific isotopes
    rxns : Tuple[int..]
        Reaction numbers
    mxs : collections.deque of MicroXsVector
        Microscopic cross sections over time
    time : collections.deque of float
        Time points such that ``mxs[j]`` was generated at point ``time[j]``
    order : int
        Fitting order

    See Also
    --------
    * :meth:`fromMicroXsVector` - helper construction method

    """

    # TODO Hide / control access to attributes that shouldn't be touched

    __slots__ = ("zai", "zptr", "rxns", "mxs", "time", "order", "_coeffs")

    def __init__(
        self, zai, zptr, rxns, mxs=None, time=None, maxlen=3, order=1, assumeSorted=True
    ):
        self.zai = zai
        self.zptr = zptr
        self.rxns = rxns

        if (time is not None) != (mxs is not None):
            raise ValueError("time and mxs must both be None or provided")

        if maxlen is not None:
            assert order < maxlen, (order, maxlen)
        self.order = order

        if time is None:
            self.time = collections.deque([], maxlen)
            self.mxs = collections.deque([], maxlen)
        else:
            assert len(time) == len(mxs)
            if not assumeSorted:
                self.time = collections.deque([], maxlen)
                self.mxs = collections.deque([], maxlen)
                for ix in numpy.argsort(time):
                    self.time.append(time[ix])
                    self.mxs.append(mxs[ix])
            else:
                self.time = collections.deque(time, maxlen)
                self.mxs = collections.deque(mxs, maxlen)

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
        return cls(mxsVector.zai, mxsVector.zptr, mxsVector.rxns,
                   mxs=(mxsVector.mxs, ), time=(time, ), maxlen=maxlen, order=order)

    def insert(self, time: float, mxs: MicroXsVector) -> None:
        """Insert microscopic cross sections to maintain ordering

        Parameters
        ----------
        time : float
            Value of time for this specific set of cross sections
        mxs : numpy.ndarray
            New cross sections to be loaded.

        """
        ix = bisect.bisect_left(self.time, time)
        self.time.insert(ix, time)
        self.mxs.insert(ix, mxs)
        self._coeffs = None

    def append(self, time: float, mxs: numpy.ndarray) -> None:
        """Append cross sections with no regard for ordering

        Parameters
        ----------
        time : float
            Value of time for this specific set of :class:`MicroXsVector`
        mxs : numpy.ndarray
            New cross sections to be loaded.

        """
        self.time.append(time)
        self.mxs.append(mxs)
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

        """
        if self._coeffs is None:
            self._coeffs = self._genCoeffs()
        mxs = numpy.empty(self.mxs[0].shape)
        for ix, c in enumerate(self._coeffs):
            mxs[ix] = polyval(time, c)
        return MicroXsVector(self.zai, self.zptr, self.rxns, mxs)

    def _genCoeffs(self):
        # convert (time, reaction, group) -> (reaction, time, group)
        data = numpy.array(self.mxs).transpose(1, 0, 2)
        coeffs = numpy.empty((data.shape[0], self.order + 1, data.shape[2]))
        for index, rxn in enumerate(data):
            # group, time
            coeffs[index] = polyfit(self.time, rxn, self.order, full=False)

        return coeffs
