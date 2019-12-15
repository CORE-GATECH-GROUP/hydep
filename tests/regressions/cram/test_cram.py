"""
Regression test for the CRAM solvers

Use full (~1000) isotope matrices provided from Serpent
"""

import pathlib
from warnings import warn

import numpy
from scipy.sparse import dok_matrix
import pytest

from hydep.internal.cram import Cram16Solver, Cram48Solver
from tests.regressions import config


class CramRegressionTester:
    ROOT = pathlib.Path(__file__).parent
    MATRIX_FILE = ROOT / "matrix.dat"
    ISO_FILE = ROOT / "densities.dat"
    SOLVERS = {"Cram16Solver": Cram16Solver, "Cram48Solver": Cram48Solver}

    def __init__(self):
        self.depmtx = self.readMatrixFile(self.MATRIX_FILE)
        self.deltaT, self.zai, self.n0, self.n1 = self.readIsotopicsFile(self.ISO_FILE)

    @staticmethod
    def readIsotopicsFile(isofile):
        with open(isofile, "r") as stream:
            header = stream.readline()
        deltaT = float(header.split()[3])
        zai, data0, data1 = numpy.loadtxt(isofile, unpack=True)
        return deltaT, zai.astype(int), data0, data1

    def writeIsotopicsFile(self, isofile, absolute, relative):
        data = numpy.vstack((self.zai, absolute, relative)).T
        comment = "{} {} {}".format(*data.shape, self.deltaT)
        numpy.savetxt(isofile, data, header=comment, fmt="%-6i %35.30e %35.30e")

    @staticmethod
    def readMatrixFile(mtxfile):
        rows, cols, values = numpy.loadtxt(mtxfile, unpack=True)
        with open(mtxfile, "r") as stream:
            header = stream.readline()
        nrows, ncols, nnz = (int(x) for x in header.split()[1:])
        mtx = dok_matrix((nrows, ncols), dtype=numpy.float64)

        for r, c, a in zip(rows.astype(int), cols.astype(int), values):
            mtx[r, c] = a
        return mtx.tocsr()

    def _getsolverfile(self, solver, status):
        return self.ROOT / "cram_{}_{}.dat".format(
            solver.alpha.size * 2, "good" if status else "error"
        )

    def execute(self, solver, update=False):
        n1 = solver(self.depmtx, self.n0.copy(), self.deltaT)

        absolute = n1 - self.n1

        goodfile = self._getsolverfile(solver, True)

        if update:
            self.writeIsotopicsFile(goodfile, n1, absolute)
            return True

        _dt, _zai, expN1, expAbsolute = self.readIsotopicsFile(goodfile)

        denscheck = numpy.allclose(expN1, n1)
        atolcheck = numpy.allclose(expAbsolute, absolute)

        if not (atolcheck and denscheck):
            badfile = self._getsolverfile(solver, False)
            self.writeIsotopicsFile(badfile, n1, absolute)
            # Raise a warning if one passed and the other didn't
            if atolcheck != denscheck:
                warn(
                    "CRAM{} solver exceeded allowable density{} tolerance".format(
                        solver.alpha.size, " regression" if not atolcheck else ""
                    )
                )
        return atolcheck or denscheck


@pytest.fixture(scope="module")
def cramharness():
    return CramRegressionTester()


@pytest.mark.parametrize("key", ("Cram16Solver", "Cram48Solver"))
def test_cram(cramharness, key):
    solver = cramharness.SOLVERS[key]
    assert cramharness.execute(solver, config.get("update", False))
