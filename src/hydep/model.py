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

    def findMaterials(self):
        """Yield all materials present in this and contained universes

        Parameters
        ----------
        memo : set, optional
            Set containing ids of previously visited materials. Don't
            pass unless you know what you're doing. If given, will
            be modified with :attr:`hydep.Material.id` of discovered
            materials

        Yields
        ------
        hydep.Material
            The first occurance of this material.

        """
        return self.root.findMaterials()

    def findBurnableMaterials(self):
        """Yield all burnable materials present in the problem

        Yields
        ------
        hydep.BurnableMaterial
            The first occurance of this material.

        """
        return self.root.findBurnableMaterials()


    def countBurnableMaterials(self):
        """Count all occurances of burnable materials

        Useful prior to cloning new burnable materials, so
        that volumes can be properly scaled.

        Returns
        -------
        Mapping[str, [hydep.BurnableMaterial, int]]
            Map of unique hashable IDs for unique burnable materials to
            the material and the number of instances found in the geometry.
        """
        return self.root.countBurnableMaterials()

    def differentiateBurnableMaterials(self):
        """Create new burnable materials across the geometry.

        This routine is important to create unique burnable materials
        that can be depleted using spatially correct fluxes and
        reaction rates.

        This method digs through contained universes and creates unique
        :class:`hydep.BurnedMaterial` objects and potentially new
        interior universes.

        """
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
