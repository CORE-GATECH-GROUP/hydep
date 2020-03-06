import numbers
from collections import deque

import numpy
from numpy.polynomial.polynomial import polyfit, polyval

from hydep.settings import SubSetting
from hydep.internal.timetravel import TimeTraveller
from hydep.typed import BoundedTyped


class SfvSettings(SubSetting, sectionName="sfv"):
    """Configuration for the SFV solver

    Parameters
    ----------
    modes : int, optional
        Number of higher order flux modes to use
    modeFraction : float, optional
        Fraction of allowable modes to use (0, 1]
    densityCutoff : float, optional
        Threshold density [#/b/cm] that isotopes must exceed
        in order to contribute when rebuilding macroscopic cross
        sections. Defaults to zero

    Attributes
    ----------
    modes : int or None
        Number of modes to use. A value of ``None`` will defer
        to :attr:`modeFrac`
    modeFraction : float
        Fraction of possible modes to use if :attr:`modes` is None
    densityCutoff : float
        Threshold density [#/b/cm] that isotopes must exceed
        in order to contribute when rebuilding macroscopic cross
        sections

    """
    modes = BoundedTyped("_modes", numbers.Integral, gt=0, allowNone=True)
    densityCutoff = BoundedTyped("_densityCutoff", numbers.Real, ge=0.0)

    def __init__(self, modes=None, modeFraction=1.0, densityCutoff=0):
        self.modes = modes
        self.modeFraction = modeFraction
        self.densityCutoff = densityCutoff

    @property
    def modeFraction(self):
        return self._modeFraction

    @modeFraction.setter
    def modeFraction(self, value):
        if not isinstance(value, numbers.Real):
            raise TypeError(f"Fraction of modes used must be real, not {type(value)}")
        elif not (0 < value <= 1):
            raise ValueError(
                f"Fraction of modes must satisfy 0 < frac <= 1, got {value}"
            )
        self._modeFraction = value

    def update(self, options):
        # None is allowed
        modes = options.pop("modes", False)
        fraction = options.pop("mode fraction", None)
        densityCutoff = options.pop("density cutoff", None)

        if options:
            remain = ", ".join(sorted(options))
            raise ValueError(
                "The following SFV settings were given, but do not "
                f"have corresponding settings: {remain}"
            )

        if modes is not False:
            if modes is None or isinstance(modes, numbers.Integral):
                self.modes = modes
            elif isinstance(modes, str) and modes.lower() == "none":
                self.modes = None
            else:
                self.modes = self.asPositiveInt("modes", modes)

        if fraction is not None:
            try:
                value = float(fraction)
            except ValueError as ve:
                raise TypeError(
                    f"Failed to coerce mode fraction={fraction} to float"
                ) from ve
            self.modeFraction = value

        if densityCutoff is not None:
            try:
                value = float(densityCutoff)
            except ValueError as ve:
                raise TypeError(
                    f"Failed to coerce density cutoff={densityCutoff} to float"
                ) from ve
            self.densityCutoff = value


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
