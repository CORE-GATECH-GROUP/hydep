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

    @staticmethod
    def substepUpdate(*args):
        """Nothing to do, nothing to update"""
        pass

    @staticmethod
    def execute() -> float:
        """Provided to fulfill the interface, but nothing is done

        Returns
        -------
        float
            Time required by this "solver"

        """
        return 0.0

    def processResults(self) -> TransportResult:
        """Return back the flux from :meth:`processBOS`"""
        return TransportResult(self._flux, [numpy.nan, numpy.nan])
