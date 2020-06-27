"""Tiny little class for storing and easily modifying time step information"""

from hydep.constants import SECONDS_PER_DAY


class TimeStep:
    """Class for modifying and storing time step information.

    Parameters
    ----------
    coarse : int, optional
        Current coarse time step. Defaults to zero
    substep : int, optional
        Current substep index. Defaults to None
    total : int, optional
        Current total time step index, reflecting sum
        of all coarse and substeps
    currentTime : float, optional
        Current point in calendar time [s]

    Attributes
    ----------
    coarse : int
        Current coarse time step index
    substep : int or None
        Current substep index. Will only be ``None`` if
        not actively in a substep regime, like in the
        initial preliminary stages
    total : int
        Current total time step index, including substeps
    currentTime : float
        Current point in calendar time [s]

    Examples
    --------
    >>> t = TimeStep()
    >>> t.coarse, t.substep, t.total, t.currentTime
    (0, None, 0, 0.0)
    >>> t.increment(100000)
    >>> t.coarse, t.substep, t.total, t.currentTime
    (1, None, 1, 100000.0)
    >>> t += 86400
    >>> t.coarse, t.substep, t.total, t.currentTime
    (1, 1, 2, 186400.0)

    """

    __slots__ = ("coarse", "substep", "total", "currentTime")

    def __init__(self, coarse=None, substep=None, total=None, currentTime=None):
        self.coarse = 0 if coarse is None else int(coarse)
        self.substep = None if substep is None else int(substep)
        self.total = 0 if total is None else int(total)
        self.currentTime = 0.0 if currentTime is None else float(currentTime)

    def increment(self, delta, coarse=True):
        """Advance across a coarse time step or substep of length ``delta`` [s]"""
        if coarse:
            self.substep = None
            self.coarse += 1
        else:
            self.substep = (self.substep + 1) if self.substep is not None else 1
        self.total += 1
        self.currentTime += delta

    def __iadd__(self, delta):
        """Advance one substep of length ``delta`` [s]"""
        self.increment(delta, coarse=False)
        return self

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(coarse={self.coarse}, substep={self.substep}, "
            f"total={self.total}, currentTime={self.currentTime / SECONDS_PER_DAY} [d])"
        )