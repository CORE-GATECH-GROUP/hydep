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

from scipy.sparse import dok_matrix

import hydep
from hydep.internal import (
    getZaiFromName,
    ReactionTuple,
    DecayTuple,
    parseZai,
    getIsotope,
    Isotope,
)
from hydep.internal.symbols import REACTION_MTS

__all__ = ["DepletionChain"]


class DepletionChain(tuple):
    # TODO Some OpenMC compatibility layer?
    def __new__(cls, isotopes):
        return super(DepletionChain, cls).__new__(cls, sorted(isotopes))

    def __init__(self, isotopes):
        self._indices = {isotope.zai: i for i, isotope in enumerate(self)}

    def __contains__(self, key):
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
                    rxnMt = REACTION_MTS[rxnType]
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

        return cls(isotopes)

    def find(self, name=None, zai=None):
        assert (name is not None) != (zai is not None)
        try:
            index = self.index(name if zai is None else zai)
        except IndexError:
            raise KeyError("Isotope {} not found in {}".format(
                name if zai is None else zai, self.__class__.__name__))
        return self[index]

    def index(self, key):
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
        assert isinstance(reactionRates, hydep.MicroXsVector)

        mtx = defaultdict(float)
        indices = {isotope.zai: i for i, isotope in enumerate(self)}
        if ordering is None:
            ordering = indices

        for ix, zai in enumerate(reactionRates.zai):
            start, stop = reactionRates.zptr[ix:ix + 2]
            rxns = dict(
                zip(reactionRates.rxns[start:stop], reactionRates.mxs[start:stop])
            )

            myIndex = indices.get(zai)
            columnIndex = ordering.get(zai)
            if myIndex is None or columnIndex is None:
                continue
            isotope = self[myIndex]

            # process transmutations

            for reaction in isotope.reactions:
                rate = rxns.get(reaction.mt)
                if rate is None:
                    continue
                mtx[columnIndex, columnIndex] -= rate * reaction.branch
                if reaction.mt == 18:  # fission
                    yields = fissionYields.get(isotope.zai)
                    if yields is None:
                        continue
                    for product, fyield in yields.items():
                        rowIndex = ordering.get(product)
                        if rowIndex is None:
                            continue
                        mtx[rowIndex, columnIndex] += rate * fyield

                elif reaction.target is None:
                    continue
                rowIndex = ordering.get(reaction.target.triplet)
                if rowIndex is None:
                    continue
                mtx[rowIndex, columnIndex] += rate * reaction.branch

            if isotope.decayModes:
                mtx[columnIndex, columnIndex] -= isotope.decayConstant

                for decay in isotope.decayModes:
                    if decay.target is None:
                        continue
                    rowIndex = ordering.get(decay.target.triplet)
                    mtx[rowIndex, columnIndex] += isotope.decayConstant * decay.branch

        dok = dok_matrix((len(ordering), ) * 2, dtype=reactionRates.mxs.dtype)
        dok.update(mtx)
        return dok.tocsr()
