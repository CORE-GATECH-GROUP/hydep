"""
Settings management

Intended to provide a simple interface between user provided
settings and individual solvers
"""

import copy
import os
import pathlib
import re
from collections.abc import Sequence
import typing
from abc import abstractmethod, ABCMeta
import numbers
import configparser

from hydep.typed import (
    TypedAttr,
    BoundedTyped,
    OptFile,
    PossiblePath,
    OptIntegral,
    OptReal,
)

_CONFIG_CLASSES = {"hydep": None}
_SUBSETTING_PATTERN = re.compile("^[A-Za-z][A-Za-z0-9_]*$")


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


def enforceInt(key: str, value: typing.Any, positive: bool = False):
    if isinstance(value, bool):
        raise TypeError(f"{key} must be integer, not boolean")
    elif not isinstance(value, numbers.Integral):
        raise TypeError(f"{key} must be integer, not {type(value)}")
    elif positive and not value > 0:
        raise ValueError(f"{key} must be positive integer, not {value}")


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


def asPositiveInt(key: str, value: typing.Any) -> int:
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
    candidate = asInt(key, value)
    if candidate > 0:
        return candidate
    raise ValueError(
        f"{key} must be positive integer: converted {value} to {candidate}"
    )


def makeAbsPath(p: typing.Union[pathlib.Path, str, typing.Any]) -> pathlib.Path:
    if isinstance(p, pathlib.Path):
        if p.is_absolute():
            return p
        return p.resolve()
    return pathlib.Path(p).resolve()


class SubSetting(metaclass=ABCMeta):
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


class HydepSettings:
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
    boundaryConditions : tuple of string
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

        bc = tuple(bc)

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
            self._basedir = makeAbsPath(base)
        else:
            raise TypeError("Basedir must be path-like. Cannot be none")

    @property
    def rundir(self) -> PossiblePath:
        return self._rundir

    @rundir.setter
    def rundir(self, run):
        if run is not None:
            self._rundir = makeAbsPath(run)
        else:
            self._rundir = None

    def updateAll(self, options):
        """Update settings for this instance and any subsections

        Sub-sections are expected to be found via first-level keys match
        ``hydep.<key>``. Settings for the Serpent interface should be
        in the sub-map ``options["hydep.serpent"]``. Any key not starting
        with ``hydep`` will be skipped.

        Parameters
        ----------
        option : dict of str -> {key: value}
            Two-tiered dictionary mapping settings for this and connected
            solvers / features.

        Raises
        ------
        ValueError
            If a subsection is found that does not adhere to
            the pattern ``hydep\\..*``, e.g. ``hydep.serpent`` is good,
            but ``hydep_serpent`` is not``
        AttributeError
            If a subsection would collide with a non-subsection
            attribute, e.g. :attr:`archiveOnSuccess`.

        See Also
        --------
        *. :attr:`update` - Updating rules for just this class

        """
        mainkeys = options.get(self.name)
        if mainkeys is not None:
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
            self.archiveOnSuccess = asBool("archive on success", archive)
        if depsolver is not None:
            self.depletionSolver = depsolver
        if bc is not None:
            self.boundaryConditions = bc
        if fitOrder is not None:
            self.fittingOrder = asInt("fitting order", fitOrder)
        # None is an acceptable value
        if fitPoints is not False:
            if fitPoints is None or isinstance(fitPoints, numbers.Integral):
                self.numFittingPoints = fitPoints
            elif isinstance(fitPoints, str) and fitPoints.lower() == "none":
                self.numFittingPoints = None
            else:
                self.numFittingPoints = asPositiveInt("fitting points", fitPoints)
        if unboundFit is not None:
            self.unboundedFitting = asBool("unbounded fitting", unboundFit)

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
            self.useTempDir = asBool("use temp dir", tempdir)

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


class SerpentSettings(SubSetting, sectionName="serpent"):
    """Main settings for Serpent solver

    All parameters are passed to corresponding attributes

    Attributes
    ----------
    datadir : pathlib.Path or None
        Directory of data files, like cross section libraries.
        If not given, will attempt to pull from ``SERPENT_DATA``
        environment variable. Libraries :attr:`acelib`,
        :attr:`declib`, and :attr:`nfylib` use this value if the
        initial paths do not point to files
    acelib : pathlib.Path or None
        Absolute path to cross section data file
    declib : pathlib.Path or None
        Absolute path to the Serpent decay file
    nfylib : pathlib.Path or None
        Absolute path to the Serpent neutron fission yield file
    sab : pathlib.Path or None
        Absolute path to the thermal scattering library file
    particles : int or None
        Number of particle to simulate per cycle
    generationsPerBatch : int or None
        Number of generations per batch
    active : int or None
        Number of active batches to simulate. Total number of active
        cycles will be ``active * generationsPerBatch``
    inactive : int or None
        Number of inative batches to simulate.
    seed : int or None
        Initial random number seed
    k0 : float or None
        Initial guess at multiplication factor. Defaults to 1.0, and must
        be bewteen zero and two
    executable : str or None
        Command to use when executing Serpent. Must be a valid shell
        command, e.g. ``"sss2"`` or ``"./sss2-custom"``
    omp : int or None
        Number of OMP threads to use. If initially given as ``None``, attempt
        to pull from ``OMP_NUM_THREADS`` environment variable
    mpi : int or None
        Number of MPI tasks to use

    """

    def __init__(
        self,
        # Writer settings
        datadir: OptFile = None,
        acelib: OptFile = None,
        declib: OptFile = None,
        nfylib: OptFile = None,
        sab: OptFile = None,
        particles: OptIntegral = None,
        generationsPerBatch: OptIntegral = None,
        active: OptIntegral = None,
        inactive: OptIntegral = None,
        seed: OptIntegral = None,
        k0: float = 1.0,
        # Runner settings
        executable: typing.Optional[str] = None,
        omp: OptIntegral = None,
        mpi: OptIntegral = None,
    ):
        if datadir is None:
            datadir = os.environ.get("SERPENT_DATA") or None
        self.datadir = datadir
        self.acelib = acelib
        self.declib = declib
        self.nfylib = nfylib
        self.sab = sab
        self.particles = particles
        self.generationsPerBatch = generationsPerBatch
        self.active = active
        self.inactive = inactive
        self.seed = seed
        self.k0 = k0
        self.executable = executable
        if omp is None:
            omp = os.environ.get("OMP_NUM_THREADS")
            omp = int(omp) if omp else None
        self.omp = omp
        self.mpi = mpi

    @property
    def datadir(self) -> PossiblePath:
        return self._datadir

    @datadir.setter
    def datadir(self, d: OptFile):
        if d is None:
            self._datadir = None
            return
        d = pathlib.Path(d).resolve()
        if not d.is_dir():
            raise NotADirectoryError(d)
        self._datadir = d

    def _validateLib(self, lib) -> pathlib.Path:
        lib = pathlib.Path(lib)
        if lib.is_file():
            return lib
        if self.datadir is not None:
            lib = self.datadir / lib
            if lib.is_file():
                return lib
        raise FileNotFoundError(lib)

    @property
    def acelib(self) -> PossiblePath:
        return self._acelib

    @acelib.setter
    def acelib(self, ace: OptFile):
        if ace is None:
            self._acelib = None
            return
        self._acelib = self._validateLib(ace)

    @property
    def declib(self) -> pathlib.Path:
        return self._declib

    @declib.setter
    def declib(self, dec: OptFile):
        if dec is None:
            self._declib = None
            return
        self._declib = self._validateLib(dec)

    @property
    def nfylib(self) -> PossiblePath:
        return self._nfylib

    @nfylib.setter
    def nfylib(self, nfy: OptFile):
        if nfy is None:
            self._nfylib = None
            return
        self._nfylib = self._validateLib(nfy)

    @property
    def sab(self) -> PossiblePath:
        return self._sab

    @sab.setter
    def sab(self, s: OptFile):
        if s is None:
            self._sab = None
            return
        s = pathlib.Path(s)
        if not s.is_file():
            raise FileNotFoundError(s)
        self._sab = s

    @property
    def particles(self) -> OptIntegral:
        return self._particles

    @particles.setter
    def particles(self, value: OptIntegral):
        if value is None:
            self._particles = None
            return
        enforceInt("particles", value, True)
        self._particles = value

    @property
    def active(self) -> OptIntegral:
        return self._active

    @active.setter
    def active(self, value: OptIntegral):
        if value is None:
            self._active = None
            return
        enforceInt("active", value, True)
        self._active = value

    @property
    def inactive(self) -> OptIntegral:
        return self._inactive

    @inactive.setter
    def inactive(self, value: OptIntegral):
        if value is None:
            self._inactive = None
            return
        enforceInt("inactive", value, True)
        self._inactive = value

    @property
    def generationsPerBatch(self) -> OptIntegral:
        return self._generations

    @generationsPerBatch.setter
    def generationsPerBatch(self, value: OptIntegral):
        if value is None:
            self._generations = None
            return
        enforceInt("generations per batch", value, True)
        self._generations = value

    @property
    def seed(self) -> OptIntegral:
        return self._seed

    @seed.setter
    def seed(self, value: OptIntegral):
        if value is None:
            self._seed = None
            return
        enforceInt("seed", value, True)
        self._seed = value

    @property
    def k0(self) -> OptReal:
        return self._k0

    @k0.setter
    def k0(self, value: OptReal):
        if value is None:
            self._k0 = None
            return
        if not isinstance(value, numbers.Real):
            value = float(value)
        if not (0 < value < 2):
            raise ValueError(value)
        self._k0 = value

    @property
    def executable(self) -> typing.Optional[str]:
        return self._executable

    @executable.setter
    def executable(self, exe: typing.Optional[str]):
        if exe is None:
            self._executable = None
            return
        if not isinstance(exe, str):
            raise TypeError(type(exe))
        self._executable = exe

    @property
    def omp(self) -> OptIntegral:
        return self._omp

    @omp.setter
    def omp(self, value: OptIntegral):
        if value is None:
            self._omp = None
            return
        enforceInt("omp", value, True)
        self._omp = value

    @property
    def mpi(self) -> OptIntegral:
        return self._mpi

    @mpi.setter
    def mpi(self, value: OptIntegral):
        if value is None:
            self._mpi = None
            return
        enforceInt("mpi", value, True)
        self._mpi = value

    def update(self, options: typing.Mapping[str, typing.Any]):
        """Update from a map of user supplied values

        All values map directly to attributes, with the following
        exceptions:

        *. ``"generations per batch"`` -> :attr:`generationsPerBatch`
        *. ``"thermal scattering"`` -> :attr:`sab`

        Parameters
        ----------
        options : mapping
            Keys are expected to be valid settings. Values will be
            set to attributes. Will be consumed with ``options.pop``

        Raises
        ------
        ValueError
            If any settings in ``options`` do not have a corresponding
            setting
        RuntimeError
            If a setting fails to be coerced to the expected data type

        """
        datadir = options.pop("datadir", False)
        acelib = options.pop("acelib", None)
        declib = options.pop("declib", None)
        nfylib = options.pop("nfylib", None)
        sab = options.pop("thermal scattering", None)
        particles = options.pop("particles", None)
        generations = options.pop("generations per batch", None)
        seed = options.pop("seed", None)
        active = options.pop("active", None)
        inactive = options.pop("inactive", None)
        k0 = options.pop("k0", None)
        executable = options.pop("executable", None)
        omp = options.pop("omp", False)
        mpi = options.pop("mpi", False)

        if options:
            remain = ", ".join(sorted(options))
            raise ValueError(
                f"The following Serpent settings were given, but "
                f"do not have corresponding attributes: {remain}",
            )

        if datadir is not False:
            # Check that it is None
            if datadir is None or (
                isinstance(datadir, str) and datadir.lower() == "none"
            ):
                self.datadir = None
            else:
                self.datadir = makeAbsPath(datadir)

        # libraries
        for value, dest in [
            [acelib, "acelib"],
            [declib, "declib"],
            [nfylib, "nfylib"],
            [sab, "sab"],
        ]:
            if value is None:
                continue

            if isinstance(value, pathlib.Path):
                candidate = value
            else:
                candidate = pathlib.Path(value)

            # If pointing to a file already, resolve to this
            if candidate.is_file():
                setattr(self, dest, candidate.resolve())
                continue

            # Try and resolve to datadir, if given
            if self.datadir is None:
                raise FileNotFoundError(
                    f"Cannot set {dest}={value} as file does not exist "
                    "and datadir is not configured"
                )
            candidate = self.datadir / candidate
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"Cannot set {dest}={candidate} from {value} using "
                    "target file does not exist"
                )
            setattr(self, dest, candidate)

        # Integers for particle settings, value of None skipped
        for value, dest in [
            [particles, "particles"],
            [generations, "generationsPerBatch"],
            [active, "active"],
            [inactive, "inactive"],
            [seed, "seed"],
        ]:
            if value is None:
                continue
            setattr(self, dest, asPositiveInt(dest, value))

        if k0 is not None:
            candidate = float(k0)
            if not (0 < candidate < 2):
                raise ValueError(
                    f"Cannot set k0 to {candidate} from {k0}. Must be "
                    "positive and less than 2"
                )
            self.k0 = candidate

        # Run settings
        if executable is not None:
            self.executable = executable

        if omp is not False:
            if omp is None or (isinstance(omp, str) and omp.lower() == "none"):
                raise ValueError(
                    f"Cannot set omp to None from {omp}. By default, OMP will "
                    "infer threads from OMP_NUM_THREADS"
                )
            self.omp = asPositiveInt("omp", omp)

        if mpi is not False:
            if mpi is None or (isinstance(mpi, str) and mpi.lower() == "none"):
                raise ValueError(f"Cannot set mpi to None from {mpi}")
            self.mpi = asPositiveInt("mpi", mpi)


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

    def update(self, options: typing.Mapping[str, typing.Any]):
        """Update the SFV solver

        The following keys map directly to attributes, if provided

        *. ``"modes"`` - :attr:`modes` (positive integer or none-like)
        *. ``"mode fraction"`` - :attr:`modeFraction` (float)
        *. ``"density cutoff"`` - :attr:`densityCutoff` (non-negative
           float)

        Parameters
        ----------
        options : mapping of key names to values
            Poppable map of settings and values

        Raises
        ------
        ValueError
            If keys exist in ``options`` that don't correspond
            to supported settings

        """
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
            if modes is None or (isinstance(modes, str) and modes.lower() == "none"):
                self.modes = None
            else:
                self.modes = asPositiveInt("modes", modes)

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
