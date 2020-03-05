"""
Settings management

Intended to provide a simple interface between user provided
settings and individual solvers
"""

import copy
import re
from collections.abc import Sequence
import typing
from abc import abstractmethod, ABCMeta
import numbers

from hydep.typed import TypedAttr

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
        except ValueError as ve:
            raise ve from TypeError(f"Could not coerce {key}={value} to {dtype}")

    @staticmethod
    def asInt(key: str, value: typing.Any) -> int:
        if not isinstance(value, numbers.Integral):
            if isinstance(value, numbers.Real):
                raise TypeError(f"Will not coerce {key}={value} from real to integer")
        return int(value)

    def asPositiveInt(self, key: str, value: typing.Any) -> int:
        candidate = self.asInt(key, value)
        if candidate > 0:
            return candidate
        raise ValueError(f"{key}={value} must be positive integer")


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
                "Not a valid Python name, and \".\" characters are disallowed"
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
    _ALLOWED_BC = frozenset({"reflective", "periodic", "vacuum"})

    def __init__(
            self,
            archiveOnSuccess=False,
            depletionSolver=None,
            boundaryConditions=None,
    ):
        self.archiveOnSuccess = archiveOnSuccess
        self.depletionSolver = depletionSolver
        if boundaryConditions is None:
            self._boundaryConditions = ("vacuum",) * 3
        else:
            self.boundaryConditions = boundaryConditions

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
