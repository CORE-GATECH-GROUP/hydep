import numbers
from collections.abc import Iterable
import typing

import numpy

from .universe import Universe
from .typed import TypedAttr
from hydep.internal import Boundaries

__all__ = ("Model",)


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
        Root universe for the problem.

    Attributes
    ----------
    root : hydep.lib.Universe
        Root universe for the problem.
    bounds : Optional[hydep.internal.Boundaries]
        X, Y, and Z boundaries for the root universe. A value of
        ``None`` is allowed, but implies the problem is unbounded in
        all directions. This may cause issues with downstream solvers.

    See Also
    --------
    * :meth:`hydep.lib.Universe.boundaries`
        Look into the root universe and determine size from contents.

    """

    root = TypedAttr("root", (Universe))

    def __init__(self, root):
        self.root = root
        self._bounds = None

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
            vols = self.root.countBurnableMaterials()
            for mat, counts in vols.values():
                if mat.volume is None:
                    raise AttributeError("Volume not set for {}".format(mat))
                mat.volume = mat.volume / counts

        self.root.differentiateBurnableMaterials()

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if bounds is None:
            self._bounds = None
            return

        if not isinstance(bounds, Boundaries):
            if not isinstance(bounds, Iterable):
                raise TypeError(
                    "Boundaries must be Iterable[Tuple[Real, Real]], not ".format(
                        bounds))
            bounds = Boundaries(*bounds)

        bx = self._checkBounds(bounds.x, "X")
        by = self._checkBounds(bounds.y, "Y")
        bz = self._checkBounds(bounds.z, "Z")

        self._bounds = Boundaries(bx, by, bz)

    @staticmethod
    def _checkBounds(b, dim):
        if b is None:
            return b
        fmt = dim + " dimension must be None, or (real, real), not {}"
        if not isinstance(b, Iterable):
            raise TypeError(fmt.format(type(b)))
        elif not len(b) == 2:
            raise ValueError(fmt.format(b))
        elif not all(isinstance(o, numbers.Real) for o in b):
            raise ValueError(fmt.format(b))
        elif b[0] >= b[1]:
            raise ValueError(
                "Lower bound {} greater than upper bound {}".format(b[0], b[1])
            )
        return b

    def isBounded(self, dim: typing.Optional[str]=None) -> bool:
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
            x = self._isDimensionBounded(bounds.x)
            if dim != "all":
                return x
            elif not x:
                return False

        if dim.lower() == "y" or dim == "all":
            y = self._isDimensionBounded(bounds.y)
            if dim != "all":
                return y
            elif not y:
                return False

        # Made it here, so either only checking z direction
        # or checking all directions with x and y being bounded
        # Z-direction doesn't need to be bounded for 2D analysis

        if dim == "all":
            return True

        return self._isDimensionBounded(bounds.z)

    @staticmethod
    def _isDimensionBounded(bounds):
        if bounds is None or None in bounds:
            return False
        return not numpy.isinf(bounds).any()
