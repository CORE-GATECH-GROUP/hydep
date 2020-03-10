"""
Simple reduced order solver.

More of a no-op, in that it doesn't actually
perform a flux solution
"""

import numpy

from hydep.internal.features import FeatureCollection
from hydep.internal import TransportResult
from .lib import ReducedOrderSolver


class SimpleROSolver(ReducedOrderSolver):
    """The simplest reduced order flux solution where nothing happens"""

    needs = FeatureCollection()

    def __init__(self):
        self._flux = None

    def processBOS(self, txResult, _timestep, _power):
        """Store flux from a high fidelity transport solution"""
        self._flux = txResult.flux

    def substepSolve(self, *args, **kwargs):
        """Return the beginning-of-step flux with no modifications

        Returns
        -------
        hydep.internal.TransportResult
            Transport result with the flux provided in :meth:`processBOS`
        """
        return TransportResult(self._flux, [numpy.nan, numpy.nan], runTime=numpy.nan)
