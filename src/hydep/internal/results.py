"""
Classes for passing results from transport and depletion
"""

from collections.abc import Sequence, Mapping
import numbers

import numpy
import scipy.sparse

from .microxs import MicroXsVector


class TransportResult:
    """Result from any transport simulation

    Each :class:`hydep.TransportSolver`
    is expected to fill at least :attr:`flux`, :attr:`keff`,
    and :attr:`runTime`. Other attributes should be set
    depending on what is needed by the reduced order solution.

    Parameters
    ----------
    flux : Iterable[Iterable[float]]
        Local flux in each burnable material scaled to problem-specific
        values, e.g. correct power. Expected to be ordered such that
        ``flux[i][g]`` is the flux in energy group ``g`` in burnable
        region ``i``
    keff : Iterable[float]
        Multiplication factor and possible absolute uncertainty.
        Presented as ``[k, unc]``. If no uncertainty is computed,
        use :type:`numpy.nan`
    runTime : Optional[float]
        If given, pass to :attr:`runTime`
    macroXS : Optional[Sequence]
        If given, pass to :attr:`macroXS`
    fmtx : scipy.sparse.csr_matrix, optional
        If given, pass to :attr:`fmtx`
    microXS : Optional[Sequence]
        If given, pass to :attr:microXS`

    Attributes
    ----------
    flux : numpy.ndarray
        Local flux in each burnable material scaled to problem-specific
        values, e.g. correct power. ``flux[i, g]`` is the ``g``-th group
        flux in burnable region ``i``.
    keff : Tuple[float, float]
        Multiplication factor and absolute uncertainty.
    runTime : Union[float, None]
        Total walltime [s] used by solution
    macroXS : Sequence[Mapping[str, Iterable[float]]]
        Homogenized macroscopic cross sections in each burnable
        region. The mapping at ``macroXS[i]`` corresponds to
        region ``i``, and maps the names of cross sections to
        vectors of their expected values, e.g.
        ``{"abs": [siga_1, siga_2, ..., siga_G]}``
    fmtx : scipy.sparse.csr_matrix or None
        Fission matrix such that ``fmtx[i, j]`` describes the
        expected number of fission neutrons born in burnable
        region ``j`` due to a fission event in burnable region
        ``i``
    microXS : Sequence of hydep.internal.MicroXsVector or None
        Microscopic cross sections in each burnable region.

    """

    __slots__ = ("_flux", "_keff", "_runTime", "_macroXS", "_fmtx", "_microXS")

    def __init__(self, flux, keff, runTime=None, macroXS=None, fmtx=None, microXS=None):
        self.flux = flux
        self.keff = keff
        self.runTime = runTime
        self.macroXS = macroXS
        self.fmtx = fmtx
        self.microXS = microXS

    @property
    def flux(self):
        return self._flux

    @flux.setter
    def flux(self, value):
        f = numpy.ascontiguousarray(value, dtype=float)
        if len(f.shape) != 2:
            raise ValueError("Expected flux to be 2D array, got {}".format(f.shape))
        self._flux = f

    @property
    def keff(self):
        return self._keff

    @keff.setter
    def keff(self, value):
        k, u = value
        if not isinstance(k, numbers.Real):
            raise TypeError("Keff must be real, not {}".format(type(k)))
        if not isinstance(u, numbers.Real):
            raise TypeError("Uncertainty on keff must be real, not {}".format(type(u)))
        self._keff = k, u

    @property
    def runTime(self):
        return self._runTime

    @runTime.setter
    def runTime(self, t):
        if t is None:
            self._runTime = None
            return
        if not isinstance(t, numbers.Real):
            raise TypeError("Runtime must be real, not {}".format(type(t)))
        self._runTime = t

    @property
    def macroXS(self):
        return self._macroXS

    @macroXS.setter
    def macroXS(self, xs):
        if xs is None:
            self._macroXS = None
            return

        if not isinstance(xs, Sequence):
            raise TypeError(
                "MacroXS must be sequence of mappings, not {}".format(type(xs))
            )
        for index, item in enumerate(xs):
            if not isinstance(item, Mapping):
                raise TypeError(
                    "All items in {}.macroXS must be Mapping. Found {} at {}".format(
                        self.__class__.__name__, type(item), index
                    )
                )
        self._macroXS = xs

    @property
    def fmtx(self):
        return self._fmtx

    @fmtx.setter
    def fmtx(self, value):
        if value is None:
            self._fmtx = None
            return

        if isinstance(value, scipy.sparse.csr_matrix):
            array = value
        elif scipy.sparse.issparse(value):
            array = value.tocsr()
        else:  # try as numpy array
            array = scipy.sparse.csr_matrix(numpy.asarray(value), dtype=float)
        if len(array.shape) != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(
                "Fission matrix must be set with a square 2D array, "
                "got {}".format(array.shape))
        self._fmtx = array

    @property
    def microXS(self):
        return self._microXS

    @microXS.setter
    def microXS(self, value):
        if value is None:
            self._microXS = None
            return

        if not isinstance(value, Sequence):
            raise TypeError("microXS must be sequence of MicroXSVector not {}".format(
                type(value)))
        for item in value:
            if not isinstance(item, MicroXsVector):
                raise TypeError("microXS must be sequence of {}, found {}".format(
                    type(item)))
        self._microXS = value
