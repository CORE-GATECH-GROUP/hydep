import numbers
from collections.abc import Sequence, Iterable

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
    root : hydep.Universe
        Root universe for the problem.

    Attributes
    ----------
    root : hydep.Universe
        Root universe for the problem.
    bounds : Optional[hydep.internal.Boundaries]
        X, Y, and Z boundaries for the root universe. A value of
        ``None`` is allowed, but implies the problem is unbounded in
        all directions. This may cause issues with downstream solvers.

    See Also
    --------
    * :meth:`hydep.Universe.boundaries`
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
        fmt = dim + " dimension must be None, or Tuple[Real, Real], not {}"
        if not isinstance(b, Sequence):
            raise TypeError(fmt.format(dim.upper(), type(b)))
        elif not len(b) == 2:
            raise ValueError(fmt.format(dim.upper(), b))
        elif not all(isinstance(o, numbers.Real) for o in b):
            raise ValueError(fmt.format(dim.upper(), b))
        elif b[0] >= b[1]:
            raise ValueError(
                "Lower bound {} greater than upper bound {}".format(b[0], b[1])
            )
        return b
