"""
Class for handling vectorized microscopic cross sections
"""
import bisect
import collections
import numbers

import numpy
from numpy.polynomial.polynomial import polyfit, polyval


class MicroXsVector:

    __slots__ = ("zai", "zptr", "rxns", "mxs")

    def __init__(self, zai, zptr, rxns, mxs):
        self.zai = zai
        self.zptr = zptr
        self.rxns = rxns
        self.mxs = mxs

    @classmethod
    def fromLongFormVectors(cls, zaiVec, rxnVec, mxs, assumeSorted=True):

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

        return cls(
            numpy.array(zai, dtype=int),
            numpy.array(zptr, dtype=int),
            rxnVec.copy(),
            mxs.copy(),
        )

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
        new = self.__class__(self.zai, self.zptr, self.rxns, self.mxs)
        new *= scalar
        return new

    def __rmul__(self, scalar):
        return self * scalar


class TemporalMicroXs:
    """Microscopic cross sections over time

    ``maxlen`` defaults to 3 to support cross sections
    from previous, current, and predicted points in time
    """

    __slots__ = ("zai", "zptr", "rxns", "mxs", "time", "order")

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

    def insert(self, time, mxs):
        ix = bisect.bisect_left(self.time, time)
        self.time.insert(ix, time)
        self.mxs.insert(ix, mxs)

    def push(self, time, mxs):
        self.time.append(time)
        self.mxs.append(mxs)

    def at(self, time):
        # TODO Store mxs as array instead of in queue?
        # self.at() will be called at least twice for every coarse step
        # while self.push will be called once per coarse step
        p = polyfit(self.time, self.mxs, self.order)
        mxs = polyval(time, p)
        return MicroXsVector(self.zai, self.zptr, self.rxns, mxs)
