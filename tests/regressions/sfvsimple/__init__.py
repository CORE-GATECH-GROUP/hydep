import math

import numpy
import scipy.sparse
import pytest
from hydep.internal import TransportResult

from tests.regressions import loadSparseMatrix, dumpSparseMatrix, CompareBase


class SfvDataHarness:

    ixSiga0 = 0
    ixNsf0 = 1
    ixSiga1 = 2
    ixSigf1 = 3
    ixPhi0 = 4
    ixPhi1 = 5
    ixNubar1 = 6
    ixKappaSigf1 = 7
    numIndexes = 8
    floatFormat = "%.7E"
    intFormat = "%5d"

    def __init__(self, xsdata, keff0, fmtx):
        self.keff0 = keff0
        self._xsdata = numpy.asfortranarray(xsdata)
        self.fmtx = fmtx

    @property
    def siga0(self) -> numpy.ndarray:
        return self._xsdata[self.ixSiga0]

    @property
    def nsf0(self) -> numpy.ndarray:
        return self._xsdata[self.ixNsf0]

    @property
    def siga1(self) -> numpy.ndarray:
        return self._xsdata[self.ixSiga1]

    @property
    def sigf1(self) -> numpy.ndarray:
        return self._xsdata[self.ixSigf1]

    @property
    def phi0(self) -> numpy.ndarray:
        return self._xsdata[self.ixPhi0]

    @property
    def phi1(self) -> numpy.ndarray:
        return self._xsdata[self.ixPhi1]

    @property
    def nubar1(self) -> numpy.ndarray:
        return self._xsdata[self.ixNubar1]

    @property
    def kappaSigf1(self) -> numpy.ndarray:
        return self._xsdata[self.ixKappaSigf1]

    def toTransportResult(self) -> TransportResult:
        res = TransportResult(
            self.phi0.copy(), [self.keff0, numpy.nan], fmtx=self.fmtx,
        )

        macroXS = []
        for siga, nsf, nubar in zip(self.siga0, self.nsf0, self.nubar1):
            # Use nubar 1 and constant extrapolation to hit exact nubar
            # at substep point
            macroXS.append({"abs": [siga], "nsf": [nsf], "nubar": [nubar]})

        res.macroXS = macroXS
        return res

    def dump(self, xsfile, fmtxfile):
        numpy.savetxt(
            xsfile, self._xsdata, fmt=self.floatFormat, header="k %.7f" % self.keff0,
        )

        dumpSparseMatrix(
            fmtxfile,
            self.fmtx.tocoo(),
            floatFormat=self.floatFormat,
            intFormat=self.intFormat,
        )

    @classmethod
    def fromDataFiles(cls, xsfile, fmtxfile):
        with open(xsfile, "r") as s:
            keff0 = float(s.readline().split()[2])
            xsdata = numpy.loadtxt(s)
        with open(fmtxfile, "r") as stream:
            fmtx = loadSparseMatrix(stream).tocsr()
        return cls(xsdata, keff0, fmtx)

    @classmethod
    def updatePieceWise(
        cls,
        xsfile,
        fmtxfile,
        keff0,
        siga0,
        nsf0,
        siga1,
        sigf1,
        phi0,
        phi1,
        nubar1,
        kappaSigf1,
        fmtx,
    ):
        harness = cls.fromPieceWise(
            keff0, siga0, nsf0, siga1, sigf1, phi0, phi1, nubar1, kappaSigf1, fmtx,
        )
        harness.dump(xsfile)

    @classmethod
    def fromPieceWise(
        cls, keff0, siga0, nsf0, siga1, sigf1, phi0, phi1, nubar1, kappaSigf1, fmtx
    ):
        shapes = {
            numpy.shape(v)
            for v in [siga0, nsf0, siga1, sigf1, phi0, phi1, nubar1, kappaSigf1]
        }
        assert len(shapes) == 1
        shape = shapes.pop()
        assert len(shape) == 1 or shape.count(1) == len(shape) - 1, shape

        data = numpy.empty((cls.numIndexes, math.prod(shape)), order="f")

        data[cls.ixSiga0, :] = siga0
        data[cls.ixNsf0, :] = nsf0
        data[cls.ixSiga1, :] = siga1
        data[cls.ixSigf1, :] = sigf1
        data[cls.ixPhi0, :] = phi0
        data[cls.ixPhi1, :] = phi1
        data[cls.ixNubar1, :] = nubar1
        data[cls.ixKappaSigf1, :] = kappaSigf1

        if not scipy.sparse.issparse(fmtx):
            fmtx = scipy.sparse.csr_matrix(numpy.asarray(fmtx))
        elif not scipy.sparse.isspmatrix_csr(fmtx):
            fmtx = fmtx.tocsr()

        return cls(data, keff0, fmtx)


class SfvComparator(CompareBase):
    """
    Class for fetching, updating, and testing SFV cases

    Parameters
    ----------
    datadir : str or pathlib.Path
        Directory for a specific case. Will read from reference files
        and write failure files in this directory
    harness : SfvDataHarness
        Macroscopic cross sections and other values from reference case

    Attributes
    ----------
    datadir : pathlib.Path
        Directory for a specific case. Will read from reference files
        and write failure files in this directory
    floatFormat : str
        String used to format a single floating point value. Passed
        to various routines like :func:`numpy.savetxt`
    intFormat : str
        String used to format integers

    """

    floatFormat = "% .7E"

    def __init__(self, datadir, harness):
        super().__init__(datadir)
        self.harness = harness

    def main(self, siga1, sigf1, nubar1, phi1, kappaSigf1):
        super().main(siga1, sigf1, nubar1, phi1, kappaSigf1)

    def _computeDifferences(self, key, value):
        refData = getattr(self.harness, key)
        out = numpy.empty((2, refData.size))
        out[0] = value - refData

        zi = refData < 1e-16
        nzi = ~zi
        out[1, zi] = out[0, zi]
        out[1, nzi] = (out[0, nzi] * 100) / refData[nzi]

        return out

    def update(self, siga1, sigf1, nubar1, phi1, kappaSigf1):
        for qty, value in (
            ("siga1", siga1),
            ("sigf1", sigf1),
            ("nubar1", nubar1),
            ("phi1", phi1),
            ("kappaSigf1", kappaSigf1),
        ):
            diffs = self._computeDifferences(qty, value).T
            numpy.savetxt(
                self.getPathFor(qty, "reference"),
                diffs,
                fmt=self.floatFormat,
                header="{} 2\nAbsolute Error, Relative Error %".format(len(value),),
            )

    def compare(self, siga1, sigf1, nubar1, phi1, kappaSigf1):
        fails = {}
        # TODO Pull from locals?
        for qty, value in (
            ("siga1", siga1),
            ("sigf1", sigf1),
            ("nubar1", nubar1),
            ("phi1", phi1),
            ("kappaSigf1", kappaSigf1),
        ):
            actual = self._computeDifferences(qty, value)
            expected = numpy.loadtxt(self.getPathFor(qty, "reference"), unpack=True)

            if not actual == pytest.approx(expected):
                fails[qty] = expected
                numpy.savetxt(
                    self.getPathFor(qty, "fail"),
                    expected.T,
                    fmt=self.floatFormat,
                    header="{} 2\nAbsolute Error, Relative Error %".format(len(value)),
                )
        return sorted(fails)
