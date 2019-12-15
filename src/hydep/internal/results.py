"""
Classes for passing results from transport and depletion
"""

import numpy

# TODO Standardize the additional data, e.g. fission matrix, macro xs, micro xs


class TransportResult:
    """Result from high-fidelity transport simulation

    Each :class:`hydep.HighFidelitySolver`
    is expected to fill at least :attr:`flux`, :attr:`keff`,
    and :attr:`runTime`. Other attributes should be set
    depending on what is needed by the reduced order solution.

    Parameters
    ----------
    flux : numpy.ndarray
        Local flux in each burnable material scaled to problem-specific
        values, e.g. correct power
    keff : iterable of float
        Multiplication factor and possible absolute uncertainty.
        Presented as ``[k, unc]``. If no uncertainty is computed,
        use :type:`numpy.nan`
    kwargs :
        Optional data needed by reduced order solver as dictated
        by hooks.

    Attributes
    ----------
    flux : numpy.ndarray
        Local flux in each burnable material scaled to problem-specific
        values, e.g. correct power
    keff : iterable of float
        Multiplication factor and absolute uncertainty.
    runTime : float
        Total walltime [s] used by solution

    """

    def __init__(self, flux, keff, **kwargs):
        self.flux = flux
        self.keff = keff
        for key, value in kwargs.items():
            setattr(self, key, value)
