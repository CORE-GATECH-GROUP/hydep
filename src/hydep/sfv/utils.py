import numbers
from collections import deque

import numpy
from numpy.polynomial.polynomial import polyfit, polyval

from hydep.internal.timetravel import TimeTraveller


class NubarPolyFit(TimeTraveller):
    """Class for performing polynomial extrapolation on nubar

    .. note::

        One group only, provided in each burnable material

    Parameters
    ----------
    maxlen : int, optional
        Maximum number of time points to retain
    order : int, optional
        Order for the polynomial fitting

    Attributes
    ----------
    times : tuple of float
        Copy of the underlying times used.
    nubar : tuple of numpy.ndarray
        Copy of stored nubar values
    order : int
        Read-only polynomial fitting order

    """

    __slots__ = TimeTraveller.__slots__ + ("_order", "_nubar", "_coeffs")

    def __init__(self, maxlen=3, order=1):
        assert isinstance(order, numbers.Integral) and order >= 0
        if maxlen is not None:
            assert isinstance(maxlen, numbers.Integral) and order < maxlen
        super().__init__(maxlen=maxlen)
        self._order = order
        self._nubar = deque(maxlen=maxlen)
        self._coeffs = None

    @property
    def order(self):
        return self._order

    @property
    def nubar(self):
        return tuple(self._nubar)

    def _validate(self, item):
        data = numpy.asarray(item)
        if len(data.shape) > 1:
            assert data.size == numpy.prod(data.shape), data.shape
            data.resize(data.size)
        return data

    def __getitem__(self, index: int) -> numpy.ndarray:
        """Return the nubar data stored at ``time[index]``"""
        return self._nubar[index]

    def __setitem__(self, index: int, nubar: numpy.ndarray):
        """Overwrite nubar data stored at index ``index``"""
        self._nubar[index] = self._validate(nubar)
        self._coeffs = None

    def _insert(self, index: int, value: numpy.ndarray):
        self._nubar.insert(index, self._validate(value))
        self._coeffs = None

    def _append(self, value: numpy.ndarray):
        self._nubar.append(self._validate(value))
        self._coeffs = None

    def _extend(self, values):
        valid = [self._validate(v) for v in values]
        self._nubar.extend(valid)
        self._coeffs = None

    def __call__(self, time: float) -> numpy.ndarray:
        """Return nubar across burnable regions at a specific time

        Apply polynomial fitting using previous number of nubar values

        Parameters
        ----------
        time : float
            Point in calendar time to evaluate

        Returns
        -------
        numpy.ndarray
            Homogenized "macroscopic" nu-bar data in each burnable region

        """
        return super().__call__(time)

    def _evaluate(self, time: float) -> numpy.ndarray:
        if self._coeffs is None:
            if len(self) == 0:
                raise AttributeError("Cannot fit using zero points")
            order = min(len(self) - 1, self._order)
            self._coeffs = polyfit(self._times, numpy.array(self._nubar), order)
        return polyval(time, self._coeffs)
