"""
Classes for interacting with microscopic cross sections and reaction rates

"""
import typing
import bisect


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
    >>> xs.index(922350, 102)
    2
    >>> xs[-1]
    (922380, 18)
    >>> xs.getZaiStart(922380)
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

    @property
    def zais(self):
        return self._zais

    @property
    def rxns(self):
        return self._rxns

    @property
    def zptr(self):
        return self._zptr

    def index(self, zai: int, rxn: int) -> int:
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
        zix = self.getZaiStart(zai)
        start, end = self.zptr[zix : zix + 2]
        rxns = self.rxns[start:end]
        for offset, r in enumerate(rxns):
            if r == rxn:
                return start + offset
        raise ValueError(f"Reaction {rxn} for {zai} not found")

    def getZaiStart(self, zai: int) -> int:
        """Return the index in :attr:`zais` for a given ZAI

        Parameters
        ----------
        zai : int
            Isotope ZAI of interest

        Returns
        -------
        int
            Index in :attr:`zais` such that
            ``x.zais[x.getZaiStart(z)] == z``

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
        zix = self.getZaiStart(zai)
        start, end = self._zptr[zix : zix + 2]
        return (
            (rxn, start + ix) for ix, rxn in enumerate(self.rxns[start:end])
        )
