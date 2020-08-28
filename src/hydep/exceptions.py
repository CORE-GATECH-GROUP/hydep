class GeometryError(Exception):
    """Raised indicating a failure in building geometry"""


class IncompatibilityError(Exception):
    """Error raised when two solvers cannot be coupled"""


class FailedSolverError(Exception):
    """Indicate that a solver has failed"""


class NegativeDensityError(RuntimeError):
    """Indicate that sufficiently negative densities were obtained"""


class NegativeDensityWarning(RuntimeWarning):
    """Warning that sufficiently negative densities were obtained"""


class DataWarning(UserWarning):
    """Warning that non-ideal behavior has been found in nuclear data"""


class DataError(Exception):
    """Raised if there is an issue with nuclear data, e.g. missing libraries"""
