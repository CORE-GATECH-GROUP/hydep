class GeometryError(Exception):
    """Raised indicating a failure in building geometry"""


class IncompatibityError(Exception):
    """Error raised when two solvers cannot be coupled"""


class FailedSolverError(Exception):
    """Indicate that a solver has failed"""
