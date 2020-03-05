"""
Settings management

Intended to provide a simple interface between user provided
settings and individual solvers
"""

from collections.abc import Sequence
import typing
from hydep.typed import TypedAttr


class HydepSettings:
    """Main setting configuration with validation

    Parameters
    ----------
    archiveOnSuccess : bool, optional
        Intial value for :attr:`archiveOnSuccess`. Default: ``False``
    depletionSolver : str, optional
        Initla value for :attr:`depletionSolver`. Default: ``"cram16"``
    boundaryConditions : str or iterable of str, optional
        Initial value for :attr:`boundaryConditions`. Default is
        vacuum in x, y, and z direction

    Attributes
    ----------
    archiveOnSuccess : bool
        True if solvers should retain any temporary files even during
        success
    depletionSolver : str
        String indicating which depletion solver to use.
    boundaryConditions : iterable of string
        Three valued list or iterable indicating X, Y, and Z
        boundary conditions

    """

    _name = "hydep"
    archiveOnSuccess = TypedAttr("_archiveOnSuccess", bool)
    depletionSolver = TypedAttr("_depletionSolver", str)
    _ALLOWED_BC = frozenset({"reflective", "periodic", "vacuum"})

    def __init__(
            self,
            archiveOnSuccess=False,
            depletionSolver=None,
            boundaryConditions=None,
    ):
        self.archiveOnSuccess = archiveOnSuccess
        self.depletionSolver = "cram16" if depletionSolver is None else depletionSolver
        if boundaryConditions is None:
            self._boundaryConditions = ("vacuum",) * 3
        else:
            self.boundaryConditions = boundaryConditions

    def asBool(self, key: str, value: typing.Union[str, bool]) -> bool:
        """Coerce a key to boolean"""
        if isinstance(value, bool):
            return value
        elif isinstance(value, int):
            if value == 0:
                return False
            if value == 1:
                return True
        elif isinstance(value, str):
            if value.lower() in {"1", "yes", "y", "true"}:
                return True
            if value.lower() in {"0", "no", "n", "false"}:
                return False

        raise TypeError(f"Could not coerce {key}={value} to boolean")

    @staticmethod
    def asType(dtype: type, key: str, value: str):
        try:
            return dtype(value)
        except ValueError:
            raise TypeError(f"Could not coerce {key}={value} to {dtype}")

    @property
    def name(self):
        return self._name

    @property
    def boundaryConditions(self):
        return self._boundaryConditions

    @boundaryConditions.setter
    def boundaryConditions(self, bc):
        self._boundaryConditions = self._validateBC(bc)

    def _validateBC(self, bc):
        if isinstance(bc, str):
            bc = bc.split()
        if not isinstance(bc, Sequence):
            raise TypeError(
                f"Boundary condition must be sequence of string, not {type(bc)}"
            )
        if len(bc) not in {1, 3}:
            raise ValueError(
                f"Number of boundary conditions must be 1 or 3, not {len(bc)}"
            )

        for b in bc:
            if b not in self._ALLOWED_BC:
                raise ValueError(
                    f"Boundary conditions {b} not valid. Must be one of "
                    f"{', '.join(sorted(self._ALLOWED_BC))}",
                )

        if len(bc) == 1:
            return bc * 3
        return bc

    def update(self, options: typing.Mapping[str, typing.Any]):
        """Update attributes using a dictionary

        Allowed keys and value types

        *. ``"archive on success"`` : boolean - update
           :attr:`archiveOnSuccess``
        *. ``"depletion solver"`` : string - update
           :attr:`depletionSolver`
        *. ``"boundary conditions"`` : string or iterable of string
           - update :attr:`boundaryConditions`

        Parameters
        ----------
        options : dict of str to object
            User-friendly strings describing attributes, like from a config
            file. Values must be able to be coerced to the expected data
            types. Will be consumed in-place as keys are removed.

        Raises
        ------
        ValueError
            If any options do not have a corresponding attribute

        """
        archive = options.pop("archive on success", None)
        depsolver = options.pop("depletion solver", None)
        bc = options.pop("boundary conditions", None)

        if options:
            raise ValueError(
                f"Not all {self.name} setting processed. The following did not "
                f"have a corresponding setting: {', '.join(options)}"
            )

        if archive is not None:
            self.archiveOnSuccess = self.asBool("archive on success", archive)
        if depsolver is not None:
            self.depletionSolver = depsolver
        if bc is not None:
            self.boundaryConditions = bc
