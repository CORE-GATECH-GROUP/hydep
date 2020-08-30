import numbers
from collections.abc import Iterable
import typing
import logging
from enum import IntEnum

import numpy

from .universe import Universe
from .typed import TypedAttr
from hydep.internal import Boundaries
from .exceptions import GeometryError

__all__ = ("Model", "Symmetry")


__logger__ = logging.getLogger("hydep.model")


class Symmetry(IntEnum):
    """Symmetry modes for model building

    Attributes
    ----------
    NONE : int
    HALF : int
    THIRD : int
    QUARTER : int
    SIXTH : int
    EIGHTH : int

    """

    NONE = 1
    HALF = 2
    THIRD = 3
    QUARTER = 4
    SIXTH = 6
    EIGHTH = 8

    @classmethod
    def fromStr(cls, value: str):
        try:
            return getattr(cls, value.upper())
        except AttributeError:
            raise ValueError(f"Symmetry option {value} not understood")

    @classmethod
    def fromInt(cls, value: int):
        for member in cls.__members__.values():
            if member == value:
                return member
        raise ValueError(f"Symmetry option {value} not understood")


class Model:
    """Representation of the entire problem geometry

    Mostly a wrapper around the :attr:`root` universe, but
    aims to provides a more general interface between the
    geometry and materials, and the transport solvers.
    Most methods for the solvers should interact directly
    with these models.

    Parameters
    ----------
    root : hydep.lib.Universe
        Root universe for the problem
    axialSymmetry : bool, optional
        Flag indicating if ``root`` has axial symmetry at ``z=0``.
        Default: False
    xySymmetry : bool, optional
        Flag indicating if ``root`` has symmetry in the XY plane.
        Incompatible with ``axialSymmetry``. Default: False
    xySymmetryType : str or int or Symmetry member, optional
        How the xy symmetry is defined. Required if ``xySymmetry``
        evalutes to True. See conditions in :meth:`applyXYSymmetry`.
        Default: :attr:`Symmetry.NONE`

    Attributes
    ----------
    root : hydep.lib.Universe
        Root universe for the problem.
    bounds : Optional[hydep.internal.Boundaries]
        X, Y, and Z boundaries for the root universe. A value of
        ``None`` is allowed, but implies the problem is unbounded in
        all directions. This may cause issues with downstream solvers.
    axialSymmetry : bool
        Read-only attribute indicating axial symmetry
    xySymmetry : Symmetry
        Read-only attribute indicating type of XY symmetry

    See Also
    --------
    * :meth:`hydep.lib.Universe.boundaries`
        Look into the root universe and determine size from contents.
    * :meth:`applyAxialSymmetry`
        Additional notes regarding axial symmetry and modeling decisions
    * :meth:`applyXYSymmetry`
        Additional notes regarding modeling constraints necessary to
        apply symmetry in the XY plane

    Notes
    -----
    To developers: :attr:`axialSymmetry` should be used to, in some way, apply
    a reflected boundary condition across the ``xy`` plane at ``z==0``.

    """

    root = TypedAttr("root", (Universe))

    def __init__(
        self,
        root,
        axialSymmetry=False,
        xySymmetry=False,
        xySymmetryType=Symmetry.NONE,
    ):
        self.root = root
        self._bounds = None
        self._xySymmetry = Symmetry.NONE
        self._axialSymmetry = False
        if axialSymmetry:
            if xySymmetry:
                raise GeometryError(
                    "Axial and XY symmetry is not supported at the moment"
                )
            self.applyAxialSymmetry()
        elif xySymmetry:
            self.applyXYSymmetry(xySymmetryType)

    @property
    def axialSymmetry(self) -> bool:
        return self._axialSymmetry

    @property
    def xySymmetry(self) -> Symmetry:
        return self._xySymmetry

    def differentiateBurnableMaterials(self, updateVolumes=True):
        """Create new burnable materials across the geometry.

        .. note::

            Calling this method more than once without changes
            to :attr:`model` is redundant. The model will be
            udpated such that each burnable material occurs only
            once after the  first call, and there is nothing more
            to perform on the second call.

        This routine is important to create unique burnable materials
        that can be depleted using spatially correct fluxes and
        reaction rates.

        This method digs through contained universes and creates unique
        :class:`hydep.BurnableMaterial` objects and potentially new
        interior universes. By default, the volume of each repeated
        :class:`hydep.BurnableMaterial` will be scaled according to
        the number of occurances found in the :attr:`model`. This
        behavior can be controlled with the ``updateVolumes``
        argument.

        Parameters
        ----------
        updateVolumes : bool
            If provided, material volumes will also be scaled by the
            number of occurances before creating new materials.

        Raises
        ------
        AttributeError
            If ``updateVolumes`` is True and
            :attr:`hydep.BurnableMaterial.volume` is not set
            for a burnable material

        """
        if updateVolumes:
            __logger__.debug("Counting and updating burnable material volumes")
            vols = self.root.countBurnableMaterials()

            for mat, counts in vols.values():
                if mat.volume is None:
                    raise AttributeError("Volume not set for {}".format(mat))
                mat.volume = mat.volume / counts

            __logger__.debug("Done.")

        __logger__.debug("Differentiating burnable materials")
        self.root.differentiateBurnableMaterials()
        __logger__.debug("Done.")

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if bounds is None:
            self._bounds = None
            return

        if isinstance(bounds, Boundaries):
            self._bounds = bounds
            return

        if not isinstance(bounds, Iterable):
            raise TypeError(
                "Boundaries must be Iterable[Tuple[Real, Real]], not "
                f"{type(bounds)}"
            )
        self._bounds = Boundaries(*bounds)

    def isBounded(self, dim: typing.Optional[str] = None) -> bool:
        """Check if the problem is bounded in all or one direction

        When checking all dimensions, the Z-axis is allowed to
        be unbounded to support 2D models. If :attr:`bounds` is None,
        then the boundaries of :attr:`root` are checked.

        Parameters
        ----------
        dim : {"all", "x", "y", "z"}, optional
            Dimension to check, case insensitive. If not provided,
            all dimensions will be checked

        Returns
        -------
        bool
            Single flag indicating if the requested dimension(s) are
            sufficiently bounded

        """
        if dim is None:
            dim = "all"
        elif not isinstance(dim, str):
            raise TypeError(f"Dimension must be string, not {dim}")

        bounds = self.root.bounds if self.bounds is None else self.bounds
        if bounds is None:
            return False

        if dim.lower() == "x" or dim == "all":
            x = not numpy.isinf(bounds.x).any()
            if dim != "all":
                return x
            elif not x:
                return False

        if dim.lower() == "y" or dim == "all":
            y = not numpy.isinf(bounds.y).any()
            if dim != "all":
                return y
            elif not y:
                return False

        # Made it here, so either only checking z direction
        # or checking all directions with x and y being bounded
        # Z-direction doesn't need to be bounded for 2D analysis

        if dim == "all":
            return True

        return not numpy.isinf(bounds.z).any()

    def applyAxialSymmetry(self):
        """Denote this model as one containing axial symmetry

        This method is designed to ease some model building,
        especially for problems that are known to axially unstable
        with respect to depletion.

        If :attr:`bounds` is not set, then the boundaries of the
        root universe will be inspected. If both are ``None`` (indicating
        unset or an issue in :meth:`hydep.Universe.boundaries`), an error
        will be raised.

        The following conditions must be met.

        1. The lower z boundary must be zero, with a finite upper z boundary.
        2. The ``xy`` plane must contain the origin

        Calling this method successfully a second time will have no effect,
        as :attr:`axialSymmetry` is set inside this method.

        .. warning::

            Altering :attr:`bounds` after calling this method is strongly
            discouraged and should be avoided at all costs.

        Raises
        ------
        hydep.GeometryError
            If any of the conditions mentioned above fail

        """
        if self.axialSymmetry:
            return
        if self.xySymmetry is not Symmetry.NONE:
            raise GeometryError("Axial and XY symmetry is not supported at this moment")

        bounds = self.bounds
        if bounds is None:
            bounds = self.root.bounds
            if bounds is None:
                __logger__.debug(
                    "Model and root universe do not have defined boundaries. "
                    "Inspecting underlying geometry"
                )
                bounds = self.root.boundaries()
                if bounds is None:
                    raise GeometryError(
                        "Model and root universe appear to be unbounded"
                    )
                self.root.bounds = bounds

        if numpy.isinf(bounds.z).any():
            raise GeometryError(
                f"Geometry is unbounded in z direction. Boundaries: {bounds.z}"
            )
        elif bounds.z.lower != 0:
            raise GeometryError(
                f"Lower z boundary must be at zero, not {bounds.z.lower}"
            )
        elif 0 not in bounds.x or 0 not in bounds.y:
            raise GeometryError(
                f"Origin not found in the xy plane: {bounds.x}, {bounds.y}"
            )

        self._bounds = Boundaries(bounds.x, bounds.y, (0, bounds.z.upper))
        self._axialSymmetry = True

    def applyXYSymmetry(self, sym):
        """Apply rotational symmetry in the XY plane

        The symmetric region is defined by a rotation starting along the
        positive x-axis and sweeping counter clockwise towards and beyond
        the positive y-axis. The geometry will be reflected

        A model of four symmetric assemblies with four unique pins in
        each assembly could be created by defining a single assembly

        .. code::

            |23
            |01
            ----

        and then applying quarter symmetry with
        :attr:`hydep.Symmetry.QUARTER` to create

        .. code::

            31|23
            20|01
            -----
            10|02
            32|13

        Parameters
        ----------
        sym : int or str or :class:`hydep.Symmetry` member
            Type of symmetry

        Raises
        ------
        hydep.GeometryError
            If this model is also configured for axial symmetry (may be
            removed in the future). If the boundaries are not set and
            cannot easily be inferred. If the point ``(x, y) = (0, 0)``
            is not found in the XY plane

        """
        if self.axialSymmetry:
            raise GeometryError(
                "Axial and XY symmetry are not supported at this moment"
            )

        if not isinstance(sym, Symmetry):
            if isinstance(sym, str):
                sym = Symmetry.fromStr(sym)
            elif isinstance(sym, numbers.Integral):
                sym = Symmetry.fromInt(sym)
            else:
                raise TypeError(f"XY symmetry type {sym} is not supported")

        if self.xySymmetry is sym:
            return

        if sym is Symmetry.NONE:
            __logger__.warn(
                "Passed %s indicating no symmetry to xySymmetry.", Symmetry.NONE.name
            )
            self._xySymmetry = Symmetry.NONE
            return

        __logger__.debug(
            "Changing %s XY symmetry from %s to %s",
            self,
            self._xySymmetry.name,
            sym.name,
        )

        bounds = self.bounds
        if bounds is None:
            bounds = self.root.bounds
            if bounds is None:
                __logger__.debug(
                    "Model and root universe do not have defined boundaries. "
                    "Inspecting underlying geometry"
                )
                bounds = self.root.boundaries()
                if bounds is None:
                    raise GeometryError(
                        f"Cannot determine boundaries on {self} and {self.root}"
                    )
                self.root.bounds = bounds

        # Check that the geometry contains the origin
        if 0 not in bounds.x:
            raise GeometryError(f"X=0 not found in {bounds.x}")

        if 0 not in bounds.y:
            raise GeometryError(f"Y=0 not found in {bounds.y}")

        self._xySymmetry = sym

        if self.bounds is None:
            self.bounds = bounds
