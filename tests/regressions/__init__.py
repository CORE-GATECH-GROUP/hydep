import math
from collections import namedtuple
import pathlib
import typing
from abc import ABC, abstractmethod

import numpy
from scipy.sparse import issparse, coo_matrix, isspmatrix_coo
import pytest

config = {"update": False}


ProblemProxy = namedtuple("ProblemProxy", "model burnable")


def dumpSparseMatrix(writable, mtx, floatfmt="%.7E", intfmt="%5d"):
    """Write a sparse or dense matrix in sparse form

    Parameters
    ----------
    writable : str or pathlib.Path or writable object
        Location to write the data. Must be understood by
        :func:`numpy.savetxt`
    mtx : numpy.ndarray or scipy.sparse.coo_matrix
        Dense or sparse matrix to be written
    floatfmt : str, optional
        Format to apply to floats, or values in the matrix
    intfmt : str, optional
        Format to apply to integers, used as row and column indexes

    See Also
    --------
    * :func:`loadSparseMatrix` - companion function

    """
    if not issparse(mtx):
        mtx = coo_matrix(mtx)
    elif not isspmatrix_coo(mtx):
        mtx = mtx.tocoo()

    numpy.savetxt(
        writable,
        numpy.transpose([mtx.row, mtx.col, mtx.data]),
        fmt=f"{intfmt} {intfmt} {floatfmt}",
        header="{} {} {}".format(mtx.nnz, *mtx.shape),
    )


def loadSparseMatrix(stream) -> coo_matrix:
    """Load a sparse matrix from a file or stream

    Parameters
    ----------
    stream : str or pathlib.Path or readable object
        Location to pass to :func:`numpy.loadtxt` that contains
        the matrix

    Returns
    -------
    scipy.sparse.coo_matrix
        Sparse representation of the matrix contained in ``stream``

    See Also
    --------
    * :func:`dumpSparseMatrix` - Companion function

    """
    if isinstance(stream, (str, pathlib.Path)):
        with open(stream) as s:
            return loadSparseMatrix(s)

    header = stream.readline()
    nnz, nrows, ncols = (int(x) for x in header.split()[1:])

    rows, cols, data = numpy.loadtxt(stream, unpack=True)

    assert rows.shape == cols.shape == data.shape == (nnz,)

    return coo_matrix((data, (rows, cols)), shape=(nrows, ncols))


class CompareBase(ABC):
    """Helper class for fetching, dumping test data

    Parameters
    ----------
    datadir : str or pathlib.Path
        Directory where test files should be found and/or written

    Attributes
    ----------
    datadir : pathlib.Path
        Directory where test files will be found / written
    floatFormat : str
        Python-2 style format string for floats. Currently ``"%.7E"``
    intFormat : str
        Python-2 style format string for integers. Currently ``"%5d"``

    """

    floatFormat = "%.7E"
    intFormat = "%5d"

    def __init__(self, datadir: typing.Union[str, pathlib.Path]):
        self.datadir = pathlib.Path(datadir)

    def getPathFor(self, qty: str, status: str):
        """Retrive a reference or failure file for a given test quantity"""
        return self.datadir / "{}_{}.dat".format(qty, status)

    def main(self, *args, **kwargs):
        """Perform the main test / update

        All args and kwargs will be passed to the abstract
        :meth:`compare` and/or :meth:`update` methods, depending
        on the pytest mode. On failure, failures will be dumped
        using :meth:`dumpFailures`

        Returns
        -------
        bool
            Status of the update or test

        Raises
        ------
        AssertionError
            If the comparisons failed

        """

        if config.get("update"):
            self.update(*args, **kwargs)
        else:
            failures = self.compare(*args, **kwargs)
            assert not failures, failures
        return True

    @abstractmethod
    def update(self, *args, **kwargs):
        """Write new reference test data"""

    @abstractmethod
    def compare(self, *args, **kwargs):
        """
        Perform a comparison against reference data

        Returns
        -------
        dict
            Dictionary with string keys indicating quantities that
            did not pass the comparison, and their corresponding
            failed values

        """

    @abstractmethod
    def dumpFailures(self, failures: dict):
        """Write failures from :meth:`compare`"""


class ResultComparator(CompareBase):
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

    def main(self, txresult):
        """Main entry point for updating or running test

        Parameters
        ----------
        txresult : hydep.internal.TransportResult
            Transport result from test. Will either be used to
            update reference data, or test against previous reference
            data.

        Returns
        -------
        bool
            Status of update or test

        Raises
        ------
        AssertionError
            If the comparison failed

        """
        return super().main(txresult)

    def update(self, txresult):
        """Update the reference files based on a new transport result"""
        self.updateKeff(txresult.keff)
        self.updateFlux(txresult.flux)
        if txresult.fmtx is not None:
            dumpSparseMatrix(
                self.getPathFor("fmtx", "reference"),
                txresult.fmtx,
                intfmt=self.intFormat,
                floatfmt=self.floatFormat,
            )

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
            self.dumpFailures(failures)
            return list(sorted(failures))
        return {}

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
            return loadSparseMatrix(stream)

    def _compareFmtx(self, fmtx):
        reference = self.referenceFmtx()
        if numpy.array_equal(fmtx.row, reference.row) and numpy.array_equal(
            fmtx.col, reference.col
        ):
            return fmtx.data == pytest.approx(reference.data)
        # Compare the full matrices to account for small values in
        # one matrix and zeros in the other
        return fmtx.A == pytest.approx(reference.A)

    def dumpFailures(self, fails):
        for key, value in fails.items():
            dest = self.getPathFor(key, "fail")
            if issparse(value):
                dumpSparseMatrix(
                    dest, value, intfmt=self.intFormat, floatfmt=self.floatFormat
                )
                continue
            numpy.savetxt(
                dest,
                value,
                fmt=self.floatFormat,
                header=" ".join(str(x) for x in value.shape),
            )
