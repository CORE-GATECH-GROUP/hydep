import numbers

import numpy

from .universe import Universe
from .typed import TypedAttr
from hydep.internal import Boundaries

__all__ = ("Model", )


class Model:
    root = TypedAttr("root", (Universe))

    def __init__(self, root):
        self.root = root
        self._bounds = False  # Special value since None is valid

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

        return self.root.differentiateBurnableMaterials()

    @property
    def bounds(self):
        if self._bounds is False:
            self.bounds = self.root.boundaries()
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if bounds is None:
            self._bounds = None
            return

        if not isinstance(bounds, Boundaries):
            assert len(bounds) == 3
            bounds = Boundaries(*bounds)

        bx, by, bz = bounds

        if bx is not None:
            assert all(isinstance(o, numbers.Real) for o in bx)
            assert bx[0] < bx[1]
        else:
            bx = (-numpy.inf, numpy.inf)

        if by is not None:
            assert all(isinstance(o, numbers.Real) for o in by)
            assert by[0] < by[1]
        else:
            by = (-numpy.inf, numpy.inf)

        if bz is not None:
            assert all(isinstance(o, numbers.Real) for o in bz)
            assert bz[0] < bz[1]
        else:
            bz = (-numpy.inf, numpy.inf)

        self._bounds = Boundaries(bx, by, bz)
