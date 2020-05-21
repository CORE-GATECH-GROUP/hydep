"""
Depletion chain data

Inspired by OpenMC - MIT Licensed
Copyright: 2011-2019 Massachusetts Institute of Technology and
OpenMC collaborators

https://docs.openmc.org
https://docs.openmc.org/en/stable/pythonapi/deplete.html
https://github.com/openmc-dev/openmc
"""

import bisect
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable
import numbers
import math

import numpy
from scipy.sparse import dok_matrix

from hydep.internal import (
    getZaiFromName,
    ReactionTuple,
    DecayTuple,
    parseZai,
    getIsotope,
    Isotope,
    FissionYieldDistribution,
    XsIndex,
)
from hydep.constants import FISSION_REACTIONS, REACTION_MT_MAP

__all__ = ["DepletionChain"]


class DepletionChain(tuple):
    """Representation of a depletion chain

    Rather than create one directly, use :meth:`fromXml`.

    Provides a ``tuple``-like interface by storing a
    sequence of :class:`hydep.internal.Isotopes`. An easier
    retrieve method is :meth:`find`, which can accept more
    varied arguments.

    Parameters
    ----------
    isotopes : iterable of hydep.internal.Isotope

    Attributes
    ----------
    zaiOrder : tuple of int
        Ordering of isotopes
    reactionIndex : hydep.internal.XsIndex
        Read-only index for describing the ordering of isotopic
        reaction cross sections and reaction rates

    """
    # TODO Some OpenMC compatibility layer?
    def __new__(cls, isotopes):
        return super(DepletionChain, cls).__new__(cls, sorted(isotopes))

    def __init__(self, isotopes):
        self._indices = {isotope.zai: i for i, isotope in enumerate(self)}
        self._zaiOrder = tuple(isotope.zai for isotope in self)
        self._reactionIndex = self._getReactionIndex()

    def __contains__(self, key):
        """Search for an isotope that matches the argument

        Parameters
        ----------
        key : str or Iterable[int] or hydep.internal.Isotope
            Isotope that may or may not exist in this chain. Strings
            represent isotope names, like ``"Xe135"`` or ``"Am242_m"``.
            Integers are treated as ``ZAI`` identifiers of the form
            ``ZZAAAI``, while a tuple or three-ple can be used to pass
            ``(z, a)`` or ``(z, a, i)`` values, respectivley.

        Returns
        -------
        bool
            If an isotope that matches ``key`` was found

        """
        if isinstance(key, str):
            key = getZaiFromName(key)
        elif isinstance(key, (Iterable, numbers.Integral)):
            try:
                key = parseZai(key)
            except (TypeError, ValueError):
                return False

        index = bisect.bisect_left(self, key)

        return index != len(self) and self[index].triplet == key

    def __repr__(self):
        return "<{} with {} isotopes at {}>".format(
            self.__class__.__name__, len(self), hex(id(self)))

    @classmethod
    def fromXml(cls, filePath):
        """Construct a chain from an OpenMC XML file

        Parameters
        ----------
        filePath : str
            File path to be processed

        Returns
        -------
        DepletionChain

        """
        isotopes = set()
        ln2 = math.log(2)

        root = ET.parse(filePath).getroot()

        for child in root:
            if child.tag != "nuclide":
                continue
            name = child.get("name")
            isotope = getIsotope(name)
            isotopes.add(isotope)

            reactions = int(child.get("reactions", 0))
            if reactions:
                for reaction in child.iter("reaction"):
                    rxnType = reaction.get("type")
                    rxnMt = REACTION_MT_MAP[rxnType]
                    qvalue = reaction.get("Q")

                    target = reaction.get("target")
                    if target == "Nothing":
                        target = None
                    elif target is not None:
                        target = getIsotope(target)
                        isotopes.add(target)

                    branchRatio = reaction.get("branching_ratio")

                    rTuple = ReactionTuple(
                        rxnMt,
                        target,
                        1.0 if branchRatio is None else float(branchRatio),
                        float(qvalue) if qvalue else None,
                    )

                    isotope.reactions.add(rTuple)

            decayModes = int(child.get("decay_modes", 0))

            if decayModes:
                isotope.decayConstant = ln2 / float(child.get("half_life"))

                for mode in child.iter("decay"):
                    decType = mode.get("type")
                    target = mode.get("target")

                    if target == "Nothing":
                        target = None
                    else:
                        target = getIsotope(target)
                        isotopes.add(target)

                    branch = mode.get("branching_ratio")

                    dTuple = DecayTuple(
                        target, decType, 1.0 if branch is None else float(branch)
                    )

                    isotope.decayModes.add(dTuple)

            fyElem = child.find("neutron_fission_yields")

            if fyElem is not None:
                isotope.fissionYields = FissionYieldDistribution.from_xml_element(fyElem)

        return cls(isotopes)

    def find(self, name=None, zai=None):
        """Return an isotope from the chain

        Parameters
        ----------
        name : Optional[str]
            Name of the isotope, such as ``"Am241_m1"``
        zai : Optional[Union[int, Iterable[int]]]
            Isotope ZAI identifier, either as ``zzaaai``
            or ``(z, a, i)``

        Returns
        -------
        hydep.internal.Isotope

        """
        assert (name is not None) != (zai is not None)
        try:
            index = self.index(name if zai is None else zai)
        except IndexError:
            raise KeyError("Isotope {} not found in {}".format(
                name if zai is None else zai, self.__class__.__name__))
        return self[index]

    def index(self, key):
        """Return the index for the isotope that matches key

        Parameters
        ----------
        key : Union[str, int, Iterable[int]]
            Name of the isotope, integer ``zzaaai`` identifier,
            or iterable ``(z, a, i)``

        Returns
        -------
        int
            Position in the chain such that
            ``chain[chain.index(x)] is chain.find(x)`` for valid
            isotope identifier ``x``

        """
        if isinstance(key, Isotope):
            isotope = key
        elif isinstance(key, str):
            isotope = getIsotope(name=key)
        else:
            isotope = getIsotope(zai=key)

        index = bisect.bisect_left(self, isotope)

        if index != len(self) and self[index] == isotope:
            return index
        raise IndexError("Could not find isotope matching {} in {}".format(
            key, self.__class__.__name__))

    def formMatrix(self, reactionRates, fissionYields, ordering=None):
        """Construct a sparse depletion matrix

        Parameters
        ----------
        reactionRates : hydep.internal.MaterialArray
            Reaction rates [#/s] for isotopes of interest. Expected
            to be indexed according to :attr:`reactionIndex`, e.g.
            ``reactionRates[ix]`` corresponds to the isotope and
            reaction located at ``self.reactionIndex[ix]``
        fissionYields : hydep.internal.FissionYield
            Fission yields mapping of the form
            ``{parentZAI: {productZAI: yield}}``
        ordering : dict of int to int, optional
            Map describing row and column indices for isotopes. If not
            provided, will sort by increasing ZAI

        Returns
        -------
        scipy.sparse.csr_matrix

        """

        mtx = defaultdict(float)
        indices = self._indices
        if ordering is None:
            ordering = indices

        for zai, columnIndex in ordering.items():

            myIndex = indices.get(zai)
            if myIndex is None:
                continue

            rxns = reactionRates.getReactions(zai, {})
            isotope = self[myIndex]

            # process transmutations

            for reaction in isotope.reactions:
                rate = rxns.get(reaction.mt)
                if rate is None:
                    continue
                mtx[columnIndex, columnIndex] -= rate * reaction.branch

                if reaction.mt in FISSION_REACTIONS:
                    yields = fissionYields.get(isotope.zai, {})
                    for product, fyield in yields.items():
                        rowIndex = ordering.get(product)
                        if rowIndex is None:
                            continue
                        mtx[rowIndex, columnIndex] += rate * fyield

                elif reaction.target is None:
                    continue
                else:
                    rowIndex = ordering.get(reaction.target.zai)
                    if rowIndex is None:
                        continue
                    mtx[rowIndex, columnIndex] += rate * reaction.branch

            if isotope.decayModes:
                mtx[columnIndex, columnIndex] -= isotope.decayConstant

                for decay in isotope.decayModes:
                    if decay.target is None:
                        continue
                    rowIndex = ordering.get(decay.target.zai)
                    if rowIndex is None:
                        continue
                    mtx[rowIndex, columnIndex] += isotope.decayConstant * decay.branch

        dok = dok_matrix((len(ordering), ) * 2, dtype=reactionRates.data.dtype)
        dict.update(dok, mtx)
        return dok.tocsr()

    @property
    def zaiOrder(self):
        return self._zaiOrder

    @property
    def reactionIndex(self) -> XsIndex:
        return self._reactionIndex

    def _getReactionIndex(self) -> XsIndex:
        """Construct an indexer to be used for reaction rates and XS"""
        zais = []
        rxns = []
        zptr = []

        for isotope in self:
            if not isotope.reactions:
                continue
            zais.append(isotope.zai)
            zptr.append(len(rxns))
            uniqrxns = {rxn.mt for rxn in isotope.reactions}
            rxns.extend(sorted(uniqrxns))

        zptr.append(len(rxns))

        return XsIndex(
            numpy.array(zais, dtype=int),
            numpy.array(rxns, dtype=int),
            numpy.array(zptr, dtype=int),
        )
