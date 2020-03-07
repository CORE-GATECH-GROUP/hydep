"""
Settings management

Intended to provide a simple interface between user provided
settings and individual solvers
"""

import copy
import pathlib
import re
from collections.abc import Sequence
import typing
from abc import abstractmethod, ABCMeta
import numbers

from hydep.typed import (
    TypedAttr,
    OptFile,
    PossiblePath,
    OptIntegral,
)

_CONFIG_CLASSES = {"hydep": None}
_SUBSETTING_PATTERN = re.compile("^[A-Za-z][A-Za-z0-9_]*$")


class ConfigMixin:
    """Mixin class for some basic type conversion"""

    @staticmethod
    def asBool(key: str, value: typing.Union[str, bool, int]) -> bool:
        """
        Coerce a key to boolean

        Parameters
        ----------
        key : str
            Name of this setting. Used in error reporting
        value : string or bool or int
            Trivial for booleans. If integer, return the corresponding
            value **only** for values of ``1`` and ``0``. If a string,
            values of ``{"1", "yes", "y", "true"}`` denote True, values
            of ``{"0", "no", "n", "false"}`` denote False. Strings are
            case-insensitive

        Raises
        ------
        TypeError
            If the conversion fails

        """
        if isinstance(value, bool):
            return value
        elif isinstance(value, int):
            if value == 0:
                return False
            if value == 1:
                return True
            raise ValueError(
                f"Could not coerrce {key}={value} to boolean. Integers "
                "must be zero or one"
            )
        elif isinstance(value, str):
            if value.lower() in {"1", "yes", "y", "true"}:
                return True
            if value.lower() in {"0", "no", "n", "false"}:
                return False

        raise TypeError(f"Could not coerce {key}={value} to boolean")

    @staticmethod
    def asType(dtype: type, key: str, value: str):
        """Convert an incoming string to a given type

        Thin wrapper around ``dtype(value)`` with a better error
        message.

        Parameters
        ----------
        dtype : type, callable
            Desired datatype or function that can produce it given ``value``
        key : str
            Name of the setting. Used only in error reporting
        value : str
            Incoming value from the configuration

        Raises
        ------
        TypeError
        """
        try:
            return dtype(value)
        except Exception as ee:
            raise TypeError(f"Could not coerce {key}={value} to {dtype}") from ee

    @staticmethod
    def asInt(key: str, value: typing.Any) -> int:
        """Coerce a value to an integer

        The following rules are applied, in order

        1. If the value is a boolean, reject
        2. If the value is an integer, return immediately
        3. If real, check process integer ratio -
           :meth:`float.as_integer_ratio`, but don't cast as ``float``
        4. Otherwise, convert to ``float`` and process integer ratio

        A value is considered an integer if the denominator of the ratio
        is positive or negative one.

        Parameters
        ----------
        key : str
            Description for this value. Used in error reporting only
        value : object
            Value that one would like to be an integer

        Returns
        -------
        int

        """
        if isinstance(value, bool):
            raise TypeError(f"Will not coerce {key}={value} from bool to integer")
        if isinstance(value, numbers.Integral):
            return value

        if isinstance(value, numbers.Real) and hasattr(value, "as_integer_ratio"):
            numer, denom = value.as_integer_ratio()
        else:
            numer, denom = float(value).as_integer_ratio()

        if denom == 1:
            return numer
        elif denom == -1:
            return -numer

        raise TypeError(f"Could not coerce {key}={value} to integer")

    def asPositiveInt(self, key: str, value: typing.Any) -> int:
        """Coerce a value to be a positive integer

        Similar rules apply as in :meth:`asInt`

        Parameters
        ----------
        key : str
            Description of the value. Used in error reporting only
        value : object
            Value that maybe can be an integer

        Returns
        -------
        int

        """
        candidate = self.asInt(key, value)
        if candidate > 0:
            return candidate
        raise ValueError(
            f"{key} must be positive integer: converted {value} to {candidate}"
        )

    @staticmethod
    def _makeAbsPath(p: typing.Union[pathlib.Path, str, typing.Any]) -> pathlib.Path:
        if isinstance(p, pathlib.Path):
            if p.is_absolute():
                return p
            return p.resolve()
        return pathlib.Path(p).resolve()


class SubSetting(ConfigMixin, metaclass=ABCMeta):
    """Abstract base class for creating solver-specific settings

    Denoted as a sub-setting, because these are used by :class:`HydepSettings`
    when a subsection is encountered. Subclasses should provide a unique
    ``sectionName`` and also implement all abstract methods, as the
    first subclass with ``sectionName`` will be found during
    :meth:`HydepSettings.updateAll`. Section names must be valid python
    expressions, and not contain any periods.

    """

    def __init_subclass__(cls, sectionName: str, **kwargs):
        if not _SUBSETTING_PATTERN.match(sectionName):
            raise ValueError(
                f"Cannot create {cls} with section name {sectionName}. "
                'Not a valid Python name, and "." characters are disallowed'
            )
        super().__init_subclass__(**kwargs)

        if sectionName in _CONFIG_CLASSES:
            reserved = ", ".join(sorted(_CONFIG_CLASSES))
            raise ValueError(
                f"Settings namespace {sectionName} already exists. Currently "
                f"reserved namespaces are {reserved}"
            )
        _CONFIG_CLASSES[sectionName] = cls

    @abstractmethod
    def update(self, options: typing.Mapping[str, typing.Any]):
        """Update given user-provided options"""


class HydepSettings(ConfigMixin):
    """Main setting configuration with validation and dynamic lookup

    Intended to be passed to various solvers in the framework. Solver
    specific settings may be included in :class:`SubSetting` instances
    that may not exist at construction, but will be dynamically
    created and assigned. For example::

    >>> h = HydepSettings()
    >>> hasattr(h, "example")
    False

    >>> class ExampleSubsection(SubSetting, sectionName="example"):
    ...    def __init__(self):
    ...        self.value = 5
    ...    def update(self, *args, **kwargs):
    ...        pass   # noop

    >>> h.example.value
    5

    Types are enforced so that downstream solvers can assume some
    constancy.

    Parameters
    ----------
    archiveOnSuccess : bool, optional
        Intial value for :attr:`archiveOnSuccess`. Default: ``False``
    depletionSolver : str, optional
        Initial value for :attr:`depletionSolver`.
    boundaryConditions : str or iterable of str, optional
        Initial value for :attr:`boundaryConditions`. Default is
        vacuum in x, y, and z direction
    fittingOrder : int, optional
        Polynomial order for fitting cross sections and other
        nuclear data over time. Defaults to one (linear) due
        to previous experience. Must be positive
    numFittingPoints : int or None, optional
        Number of points to use when fitting data. Defaults
        to three due to previous experience
    unboundedFitting : bool, optional
        True to keep indefinitely many points for cross section
        extrapolation.
    basedir : str or pathlib.Path, optional
        Directory where result files and archived files should be saved.
        Defaults to current working directory
    rundir : str or pathlib.Path, optional
        Directory where the simulation will be run if different that
        ``basedir``. Auxillary files may be written here.
    useTempDir : bool, optional
        Use a temporary directory in place of :attr:`rundir` when running
        simulations. Default is False.

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
    fittingOrder : int
        Polynomial order for fitting cross sections and other
        nuclear data over time. Must be positive
    numFittingPoints : int or None
        Number of points to use when fitting data. A value
        of ``None`` implies :attr:`unboundedFitting`
    unboundedFitting : bool, optional
        True to keep indefinitely many points for cross section
        extrapolation.
    basedir : pathlib.Path
        Directory where result files and archived files should be saved.
        If not given as an absolute path, resolves relative to the
        current working directory
    rundir : pathlib.Path or None
        Directory where the simulation will be run if different that
        ``basedir``. Auxillary files may be written here. Passing a
        value of ``None`` indicates to use the same directory as
        :attr:`basedir`. If not given as an absolute path, resolves
        relative to the current working directory
    useTempDir : bool
        Flag signalling to use a temporary directory in :attr:`rundir`
        is ``None``

    Examples
    --------

    >>> import pathlib
    >>> import os.path
    >>> pwd = pathlib.Path.cwd()
    >>> settings = HydepSettings()
    >>> settings.basedir == pwd
    True
    >>> newbase = (pwd / "base").name
    >>> newbase == "base"
    True
    >>> settings.basedir = newbase
    >>> settings.basedir.is_absolute()
    True
    >>> settings.basedir == pwd / "base"
    True

    Same resolution rules apply to :attr:`rundir`

    >>> settings.rundir = (pwd / "run").name
    >>> settings.rundir.is_absolute()
    True
    >>> settings.rundir == pwd / "run"
    True

    See Also
    --------
    *. :meth:`validate` - Performs some checks on current settings

    """

    _name = "hydep"
    archiveOnSuccess = TypedAttr("_archiveOnSuccess", bool)
    _ALLOWED_BC = frozenset({"reflective", "periodic", "vacuum"})
    unboundedFitting = TypedAttr("_unboundedFitting", bool)
    useTempDir = TypedAttr("_useTempDir", bool)

    def __init__(
        self,
        archiveOnSuccess: bool = False,
        depletionSolver: typing.Optional[typing.Any] = None,
        boundaryConditions: typing.Optional[typing.Sequence[str]] = None,
        fittingOrder: int = 1,
        numFittingPoints: int = 3,
        unboundedFitting: bool = False,
        basedir: OptFile = None,
        rundir: OptFile = None,
        useTempDir: typing.Optional[bool] = False,
    ):
        self.archiveOnSuccess = archiveOnSuccess
        self.depletionSolver = depletionSolver
        if boundaryConditions is None:
            self._boundaryConditions = ("vacuum",) * 3
        else:
            self.boundaryConditions = boundaryConditions
        self.fittingOrder = fittingOrder
        self.numFittingPoints = numFittingPoints
        self.unboundedFitting = unboundedFitting
        self.basedir = basedir or pathlib.Path.cwd()
        self.rundir = rundir
        self.useTempDir = useTempDir

    def __getattr__(self, name):
        klass = _CONFIG_CLASSES.get(name)
        if klass is None:
            raise AttributeError(
                f"No attribute nor sub-settings of {name} found on {self}"
            )
        subset = klass()
        setattr(self, name, subset)
        return subset

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

    @property
    def fittingOrder(self) -> int:
        return self._fittingOrder

    @fittingOrder.setter
    def fittingOrder(self, value):
        if value is None:
            self._fittingOrder = None
            return
        if not isinstance(value, numbers.Integral):
            raise TypeError(
                f"fitting order must be non-negative integer or None, not {value}"
            )
        elif value < 0:
            raise ValueError("fitting order cannot be negative (for now)")
        self._fittingOrder = value

    @property
    def numFittingPoints(self) -> OptIntegral:
        return self._numFittingPoints

    @numFittingPoints.setter
    def numFittingPoints(self, value):
        if value is None:
            self._numFittingPoints = None
            return
        if not isinstance(value, numbers.Integral):
            raise TypeError(f"fitting points must be positive integer, not {value}")
        elif not value > 0:
            raise ValueError(f"fitting points must be positive integer, not {value}")
        self._numFittingPoints = value

    @property
    def basedir(self) -> pathlib.Path:
        return self._basedir

    @basedir.setter
    def basedir(self, base):
        if base is not None:
            self._basedir = self._makeAbsPath(base)
        else:
            raise TypeError("Basedir must be path-like. Cannot be none")

    @property
    def rundir(self) -> PossiblePath:
        return self._rundir

    @rundir.setter
    def rundir(self, run):
        if run is not None:
            self._rundir = self._makeAbsPath(run)
        else:
            self._rundir = None

    def updateAll(self, options):
        """Update settings for this instance and any subsections

        Sub-sections are expected to be found via first-level keys match
        ``hydep.<key>``. Settings for the Serpent interface should be
        in the sub-map ``options["hydep.serpent"]``. Any key not starting
        with :attr:`self.name` will be skipped.

        Parameters
        ----------
        option : dict of str -> {key: value}
            Two-tiered dictionary mapping settings for this and connected
            solvers / features.

        Raises
        ------
        ValueError
            If no values are found under :attr:`name`. If no
            corresponding sub-section is found under ``<name>.<key>``.
        AttributeError
            If a subsection would collide with a non-subsection
            attribute, e.g. :attr:`archiveOnSuccess`.

        See Also
        --------
        *. :attr:`update` - Updating rules for just this class

        """
        mainkeys = options.get(self.name)
        if mainkeys is None:
            raise ValueError(f"No settings for {self.name} found. Refusing to advance")

        self.update(copy.copy(mainkeys))

        # Process sub-settings, but only go one level deep

        pattern = re.compile(f"^{self.name}\\.(.*)")

        for key, section in options.items():
            if not key.startswith(self.name) or key == self.name:
                continue

            match = pattern.search(key)
            if match is None:
                raise ValueError(f"Subsections must match {pattern}. Found {key}")

            name = match.groups()[0]

            if not hasattr(self, name):
                raise ValueError(f"No section found matching f{name}")

            groups = getattr(self, name)
            if not isinstance(groups, SubSetting):
                raise AttributeError(
                    f"Cannot provide a subsection {key} that matches a main setting "
                    f"{name}"
                )
            groups.update(section)

    def update(self, options: typing.Mapping[str, typing.Any]):
        """Update attributes using a dictionary

        Allowed keys and value types

        *. ``"archive on success"`` : boolean - update
           :attr:`archiveOnSuccess``
        *. ``"depletion solver"`` : string - update
           :attr:`depletionSolver`
        *. ``"boundary conditions"`` : string or iterable of string
           - update :attr:`boundaryConditions`
        *. ``"fitting order"`` : int - update :attr:`fittingOrder`
        *. ``"fitting points"`` : int - update :attr:`numFittingPoints`
        *. ``"unbounded fitting"`` : boolean - update
           :attr:`unboundedFitting`
        *. ``"basedir"`` : path-like - update :attr:`basedir`
        *. ``"rundir"`` : path-like - update :attr:`rundir`
        *. ``"use temp dir"`` : boolean - update :attr:`useTempDir`

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

        fitOrder = options.pop("fitting order", None)
        # None is an acceptable value here
        fitPoints = options.pop("fitting points", False)
        unboundFit = options.pop("unbounded fitting", None)

        # Directories
        basedir = options.pop("basedir", None)
        rundir = options.pop("rundir", False)
        tempdir = options.pop("use temp dir", None)

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
        if fitOrder is not None:
            self.fittingOrder = self.asInt("fitting order", fitOrder)
        # None is an acceptable value
        if fitPoints is not False:
            if fitPoints is None or isinstance(fitPoints, numbers.Integral):
                self.numFittingPoints = fitPoints
            elif isinstance(fitPoints, str) and fitPoints.lower() == "none":
                self.numFittingPoints = None
            else:
                self.numFittingPoints = self.asPositiveInt("fitting points", fitPoints)
        if unboundFit is not None:
            self.unboundedFitting = self.asBool("unbounded fitting", unboundFit)

        if basedir is not None:
            if isinstance(basedir, str) and basedir.lower() == "none":
                raise TypeError(f"basedir must be path-like, not {basedir}")
            else:
                self.basedir = basedir

        if rundir is not False:
            if isinstance(rundir, str) and rundir.lower() == "none":
                self.rundir = None
            else:
                self.rundir = rundir

        if tempdir is not None:
            self.useTempDir = self.asBool("use temp dir", tempdir)

    def validate(self):
        """Validate settings"""
        if self.unboundedFitting:
            if self.numFittingPoints is not None:
                raise ValueError(
                    f"Requesting unbounded fitting and {self.numFittingPoints} "
                    "points to be retained for fitting not allowed"
                )
        elif self.fittingOrder > self.numFittingPoints:
            raise ValueError(
                f"Cannot produce a {self.fittingOrder} polynomial fit with "
                f"{self.numFittingPoints} points"
            )
