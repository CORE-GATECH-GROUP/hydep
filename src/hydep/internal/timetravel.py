"""
Classes for extrapolating / interpolating quantities through time
"""

import math
from collections import deque
import typing


import numpy
from numpy.polynomial import polynomial


class TimeTraveler:
    """Store and project time-dependent array data

    Parameters
    ----------
    nsteps : int
        Maximum number of time points to hold in memory
    shape : tuple of int
        Shape of arrays that will be provided at each time point
    order : int
        Polynomial order for projection. Will use up to this order,
        depending on the number of points provided through
        :meth:`push`

    """

    def __init__(self, nsteps, shape, order):
        self._data = numpy.empty(
            (nsteps,) + tuple(shape), dtype=numpy.float64,
        )
        self._times = numpy.empty(nsteps)
        self._timeIndex = deque(maxlen=nsteps)
        self._coeffs = None
        self._order = order

    @property
    def shape(self) -> typing.Tuple[int, ...]:
        return self._data.shape

    @property
    def stacklen(self) -> int:
        return len(self._timeIndex)

    def push(self, t, data):
        """Push another set of data to be extrapolated

        Time value are expected to be pushed in order of
        increasing time.

        Parameters
        ----------
        t : float
            Point in calendar time at which ``data`` was
            generated. Units should be consistent with
            subsequent calls to :meth:`at`.
        data : numpy.ndarray
            Data generated at point ``t``

        """
        if self._timeIndex:
            if t <= self._times[self._timeIndex[-1]]:
                raise ValueError(
                    f"Current time {t} is less than maximum time "
                    f"{self._times[self._timeIndex[-1]]}"
                )
            index = (self._timeIndex[-1] + 1) % self._data.shape[0]
        else:
            index = 0
        self._data[index] = data
        self._timeIndex.append(index)
        self._times[index] = t
        self._coeffs = None

    def at(
        self, t: float, atol: typing.Optional[float] = 1e-12,
    ):
        """Project data to a new point in time

        Will make a polynomial fitting based on the number
        of current points stored and the supplied fitting
        order. If the order exceeds the maximum polynomial
        order that could be built, this maximum order will be
        used, e.g. will not use a quadratic fit until three or
        more points have been loaded.

        Parameters
        ----------
        t : float
            Point in calendar time at which cross sections are
            requested. Units should be consistent with units
            used in previous calls to :meth:`push`
        atol : float, optional
            Absolute tolerance to use when checking if ``t``
            corresponds to a currently stored point in
            time

        Returns
        -------
        numpy.ndarray
            Cross sections at the requested point in time

        Raises
        ------
        AttributeError
            If no calls to :meth:`push` have been made and
            there is no data to be projected

        """

        if not self._timeIndex:
            raise AttributeError("No data points loaded")

        # Sequential search because we should only be storing
        # a few time points, and time vector is not sorted
        for ix in self._timeIndex:
            if math.fabs(self._times[ix] - t) <= atol:
                vals = self._data[ix]
                break
        else:
            if self._coeffs is None:
                self._coeffs = polynomial.polyfit(
                    self._times[self._timeIndex],
                    self._data[self._timeIndex].reshape(
                        self.stacklen, numpy.prod(self.shape[1:])
                    ),
                    deg=min(self.stacklen - 1, self._order),
                )

            vals = polynomial.polyval(t, self._coeffs).reshape(self.shape[1:])
        return vals
