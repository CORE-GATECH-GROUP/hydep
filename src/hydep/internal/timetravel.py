"""
Classes for extrapolating / interpolating quantities through time
"""

import bisect
from abc import ABC, abstractmethod
import numbers
from collections import deque
import typing


class TimeTraveler(ABC):
    """Base class for fitting time dependent data

    Can be subclassed to perform arbitrary fitting given
    ordered points in time. Subclasses should take caution to
    maintain an identical numbering of time points and corresponding
    values. Some protections are provided through :meth:`insort`,
    :meth:`append`, and :meth:`extend`.

    If ``times`` is provided, ``values`` must be as well, and vice
    versa. Both parameters must have the same number of elements.

    Parameters
    ----------
    maxlen : int, optional
        Uses a :class:`collection.deque` under the scenes for
        storing time values. Passing ``None`` places no restrictions
        on the number of items stored. An integer value indicates that
        only ``maxlen`` values should be stored at any one point. Older
        time points will be pushed out automatically.
    times : iterable of float, optional
        Initial points in calendar time, in some consistent units like
        seconds for days, that will be used for any fitting. Must be
        sorted in an ascending order.
    values : iterable of objects, optional
        Initial values to load via :meth:`extend`.

    Attributes
    ----------
    times : tuple of float
        Copy of the underlying times used.

    See Also
    --------
    * :class:`hydep.internal.TemporalMicroXs` - Concrete class
      that uses polynomial fitting to model changes in microscopic
      reaction cross sections

    """

    __slots__ = ("_times", )

    def __init__(
        self,
        maxlen: typing.Optional[int] = None,
        times: typing.Optional[typing.Iterable[float]] = None,
        values: typing.Optional[typing.Iterable[typing.Any]] = None,
    ):
        self._times = deque(maxlen=maxlen)
        if times is not None:
            if values is None:
                raise ValueError("Cannot provide just initial times")
            self.extend(times, values)
        elif values is not None:
            raise ValueError("Cannot provide just initial values")

    @property
    def times(self):
        return tuple(self._times)

    @abstractmethod
    def __getitem__(self, index: int) -> typing.Any:
        """Return the value stored for time point ``index``"""

    @abstractmethod
    def __setitem__(self, index: int, value: typing.Any):
        """Overwrite the value stored at index ``index``"""

    def __contains__(self, time: float) -> bool:
        """Return True if time is present, otherwise False"""
        return self.find(time) is not None

    def __len__(self) -> float:
        """Number of time points currently stored"""
        return len(self._times)

    def __bool__(self) -> bool:
        """True if values are stored, otherwise False"""
        return bool(self._times)

    def index(self, time: float) -> int:
        """Return ``i`` such that ``s[i] == time``"""
        return self._times.index(time)

    def find(self, time: float) -> typing.Union[int, None]:
        """Return ``i`` such that ``s[i] == time`` or None

        Parameters
        ----------
        time : float
            Time that may or may not exist

        Returns
        -------
        index : int or None
            If ``time`` is found, return the index. Otherwise
            return ``None``

        """
        try:
            return self.index(time)
        except ValueError:
            return None

    def insort(self, time: float, value: typing.Any):
        """Insert an object such that the time ordering is consistent"""
        if not isinstance(time, numbers.Real):
            raise TypeError("Time must be numbers.Real, not {}".format(type(time)))

        ix = bisect.bisect_left(self._times, time)
        if ix == len(self):
            return self.append(time, value)

        if self._times.maxlen is None or len(self) < self._times.maxlen - 1:
            self._insert(ix, value)
            self._times.insert(ix, time)
            return

        # No point in adding to the end where it would fall off
        if ix == 0:
            return

        self.popleft()
        self._insert(ix - 1, value)
        self._times.insert(ix - 1, time)

    @abstractmethod
    def _insert(self, index: int, value: typing.Any):
        """Insert value into the collection

        Comes from :meth:`insort` where ``value`` originates
        from a specific point in time. The index is provided
        to maintain an ascending ordering of time points.

        Time will be inserted only after this call as to avoid
        a different number of times and values if the implementation
        of this method raises errors.

        Parameters
        ----------
        index : int
            Position to insert the value
        value : object
            Value to be inserted

        """

    def popleft(self) -> typing.Tuple[float, typing.Any]:
        """Remove the value that corresponds to the smallest time point"""
        v = self._popleft()
        t = self._times.popleft()
        return t, v

    @abstractmethod
    def _popleft(self):
        """Remove the value from the smallest time point

        Used by :meth:`popleft` and :meth:`insort` in some cases.
        """

    def append(self, time: float, value: typing.Any):
        """Append time points and values to the top of the deque

        Assumes that ``time`` is greater than any stored values.
        Otherwise, use :meth:`insert`
        """
        if not isinstance(time, numbers.Real):
            raise TypeError("Time must be numbers.Real, not {}".format(type(time)))
        elif self and time <= self._times[-1]:
            raise ValueError(
                "Time value {} not greater than last value {}. Use insert "
                "instead".format(time, self._times[-1])
            )
        self._append(value)
        self._times.append(time)

    @abstractmethod
    def _append(self, value: typing.Any):
        """Add the value as the most recent temporal value

        Called by :meth:`append` such that ``value`` comes from
        the greatest point in time. This point in time will
        be appended after this call, as to avoid mismatched
        number of time points and values.

        Parameters
        ----------
        value : object
            Value to be appended

        """

    def extend(
        self, times: typing.Iterable[float], values: typing.Iterable[typing.Any]
    ):
        """Add values from a collection of sorted points in time

        Times points must be increasing in value, and greater than the
        last value current stored, if present.
        """
        if len(times) != len(values):
            raise ValueError(
                "Inconsistent number of times {} and values {}".format(
                    len(times), len(values)
                )
            )
        if self:
            # check first value
            prev = self._times[len(self) - 1]
            start = 0
        else:
            if not isinstance(times[0], numbers.Real):
                raise TypeError(
                    "Non-float found at position 0: {}".format(type(times[0]))
                )
            prev = times[0]
            start = 1

        for ix, time in enumerate(times[start:], start=start):
            if not isinstance(time, numbers.Real):
                raise TypeError(
                    "Non-float found at position ix: {}".format(ix, type(times[0]))
                )
            elif time < prev:
                raise ValueError(
                    "Value {} at position {} lower than previous of {}".format(
                        ix, time, prev
                    )
                )
                prev = time

        self._extend(values)
        self._times.extend(times)

    @abstractmethod
    def _extend(self, values: typing.Iterable[typing.Any]):
        """Add a collection of values corresponding to later time points

        Called during :meth:`extend`, and useful for adding a bevy of results
        at once. Each value in ``values`` corresponds to a point in time, sorted
        to maintain an ascending order.

        Times are added after this method to avoid mismatched numbers of
        values and times.

        """

    def __call__(self, time: float) -> typing.Any:
        """Return a quantity at a given time

        Parameters
        ----------
        time : float
            Time to evaluate

        Returns
        -------
        object
            Quantity of interest at time ``time``

        """
        index = self.find(time)
        if index is not None:
            return self[index]
        return self._evaluate(time)

    @abstractmethod
    def _evaluate(self, time: float) -> typing.Any:
        """Return a value at a given time

        Called if the value of ``time`` does not exist
        in the time vector.

        """


class CachedTimeTraveler(TimeTraveler):
    """Subclass that retains a reference to previously computed values

    All input arguments are forwarded directly to the
    :class:`TimeTraveler` init method.
    """

    __slots__ = TimeTraveler.__slots__ + ("_cacheTime", "_cacheVal")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cacheTime = None
        self._cacheVal = None

    def __call__(self, time: float) -> typing.Any:
        """Return a quantity at a given time

        Parameters
        ----------
        time : float
            Time to evaluate

        Returns
        -------
        object
            Quantity of interest at time ``time``

        """
        if self._cacheTime == time:
            return self._cacheVal
        newval = super().__call__(time)
        self._cacheTime = time
        self._cacheVal = newval
        return newval
