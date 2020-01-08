import math
import pathlib

import numpy
from scipy.sparse import issparse, coo_matrix
import pytest

config = {"update": False}


class ResultComparator:
    """
    Class for fetching / comparing transport results for testing

    Parameters
    ----------
    datadir : Union[pathlib.Path, str]
        Directory for a specific case. Will read from reference files
        and write failure files in this directory

    Attributes
    ----------
    datadir : Union[pathlib.Path, str]
        Directory for a specific case. Will read from reference files
        and write failure files in this directory
    floatFormat : str
        String used to format a single floating point value. Passed
        to various routines like :func:`numpy.savetxt`
    """

    floatFormat = "%.7E"
    intFormat = "%5d"

    def __init__(self, datadir):
        self.datadir = pathlib.Path(datadir)

    def getPathFor(self, qty, status):
        """Retrive a reference or failure file for a given test quantity"""
        return self.datadir / "{}_{}.txt".format(qty, status)

    def update(self, txresult):
        """Update the reference files based on a new transport result"""
        self.updateKeff(txresult.keff)
        self.updateFlux(txresult.flux)
        if txresult.fmtx is not None:
            self._writeSparse(txresult.fmtx, self.getPathFor("fmtx", "reference"))

    def updateKeff(self, newkeff):
        """Update reference multiplication factor and absolute uncertainty"""
        fmt = " ".join([self.floatFormat] * 2) + "\n"
        with self.getPathFor("keff", "reference").open("w") as stream:
            stream.write(fmt % (newkeff[0], newkeff[1]))

    def updateFlux(self, flux):
        """Update the reference group-wise flux in each burnable region"""
        flux = numpy.asarray(flux)
        numpy.savetxt(
            self.getPathFor("flux", "reference"),
            flux,
            fmt=self.floatFormat,
            header=" ".join(map(str, flux.shape)),
        )

    def _writeSparse(self, value, where):
        if not issparse(value):
            value = coo_matrix(value)
        elif not isinstance(value, coo_matrix):
            value = value.tocoo()

        numpy.savetxt(
            where,
            numpy.transpose([value.row, value.col, value.data]),
            fmt="{i} {i} {f}".format(i=self.intFormat, f=self.floatFormat),
            header="{} {} {}".format(value.nnz, *value.shape),
        )

    def compare(self, txresult):
        """Compare results from a regression test to the reference"""
        failures = {}
        if not self._compareKeff(txresult.keff):
            failures["keff"] = numpy.array([txresult.keff])

        if not txresult.flux == pytest.approx(self.referenceFlux()):
            failures["flux"] = txresult.flux

        if txresult.fmtx is not None:
            fmtx = txresult.fmtx.tocoo()
            if not self._compareFmtx(fmtx):
                failures["fmtx"] = fmtx

        if failures:
            self._dumpfailures(failures)
            return list(sorted(failures))
        return []

    def _compareKeff(self, keff):
        refK, refU = self.referenceKeff()
        actK, actU = keff
        propUnc = math.sqrt(refU * refU + actU * actU)
        return abs(refK - actK) == pytest.approx(0, abs=propUnc)

    def referenceKeff(self):
        """Reference multiplication factor and absolute uncertainty"""
        with self.getPathFor("keff", "reference").open("r") as stream:
            line = stream.readline()
        keff, unc = (float(x) for x in line.split())
        return keff, unc

    def referenceFlux(self):
        """Reference group flux in each burnable region"""
        flux = numpy.loadtxt(self.getPathFor("flux", "reference"))
        if len(flux.shape) == 1:
            flux = flux.reshape(flux.size, 1)
        return flux

    def referenceFmtx(self):
        path = self.getPathFor("fmtx", "reference")
        with path.open("r") as stream:
            header = stream.readline()
        nnz, nrows, ncols = (int(x) for x in header.split()[1:])

        rows, cols, data = numpy.loadtxt(path, unpack=True)

        assert rows.shape == cols.shape == data.shape == (nnz,)

        return coo_matrix((data, (rows, cols)), shape=(nrows, ncols))

    def _compareFmtx(self, fmtx):
        reference = self.referenceFmtx()
        if (numpy.array_equal(fmtx.row, reference.row)
                and numpy.array_equal(fmtx.col, reference.col)):
            return fmtx.data == pytest.approx(reference.data)
        # Compare the full matrices to account for small values in
        # one matrix and zeros in the other
        return fmtx.A == pytest.approx(reference.A)

    def _dumpfailures(self, fails):
        for key, value in fails.items():
            dest = self.getPathFor(key, "fail")
            if issparse(value):
                self._writeSparse(value, dest)
                continue
            numpy.savetxt(
                dest,
                value,
                fmt=self.floatFormat,
                header=" ".join(str(x) for x in value.shape),
            )
