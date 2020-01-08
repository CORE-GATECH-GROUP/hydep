import math
import pathlib

import numpy
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

    def __init__(self, datadir):
        self.datadir = pathlib.Path(datadir)

    def getPathFor(self, qty, status):
        """Retrive a reference or failure file for a given test quantity"""
        return self.datadir / "{}_{}.txt".format(qty, status)

    def update(self, txresult):
        """Update the reference files based on a new transport result"""
        self.updateKeff(txresult.keff)
        self.updateFlux(txresult.flux)

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

    def _dumpfailures(self, fails):
        for key, value in fails.items():
            # TODO Some check for special, non-array or sparse values
            dest = self.getPathFor(key, "fail")
            numpy.savetxt(
                dest,
                value,
                fmt=self.floatFormat,
                header=" ".join(str(x) for x in value.shape),
            )
