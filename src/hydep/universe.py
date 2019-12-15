"""
Abstract base class for representing geometry that stores materials
"""

import numbers
import copy
from abc import ABC, abstractmethod

import numpy

from .materials import Material, BurnableMaterial
from hydep.typed import TypedAttr
from hydep.internal import Boundaries

__all__ = ["Universe", "InfiniteMaterial"]


class Universe(ABC):
    """Representation of a space containing materials and/or universes

    The fundamental building block for this geometry framework, the
    Universe is responsible for linking physical constructs to
    :class:`hydep.Material` objects. Base classes are responsible for
    providing methods for digging through the Universe and acting
    on sub-universes. More information is provided in the abstract
    methods

    Paramters
    ---------
    name : str, optional
        Name of this universe. Can be ``None``, and not required
        to be unique

    Attributes
    ----------
    name : str or None
        Name of this universe.
    id : str
        Unique identifier for each instance
    bounds : None or Iterable[Iterable[float]] or :class:`hydep.internal.Boundaries`
        Spatial bounds for this universe. A value of ``None`` implies
        unbounded in space.
    """

    def __init__(self, name=None):
        self.name = name
        self._bounds = False

    def __repr__(self):
        return "<{}{} at {}>".format(
            self.__class__.__name__,
            (" " + self.name) if self.name is not None else "",
            self.id,
        )

    @property
    def id(self):
        """Unique identifier for this instance"""
        return hex(id(self))

    @abstractmethod
    def findMaterials(self, memo=None):
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

    def findBurnableMaterials(self, memo=None):
        """Yield all :class:`hydep.BurnableMaterials`

        Parameters
        ----------
        memo : set, optional
            Set containing ids of previously visited materials. Don't
            pass unless you know what you're doing. If given, will
            be modified with :attr:`hydep.BurnableMaterial.id` for
            each material found.

        Yields
        ------
        hydep.BurnableMaterial
            The first occurace of this material.

        """
        memo = set() if memo is None else memo
        for mat in self.findMaterials(memo):
            if isinstance(mat, BurnableMaterial):
                yield mat

    @abstractmethod
    def countBurnableMaterials(self, memo=None):
        """Count all occurances of burnable materials

        Useful prior to cloning new burnable materials, so
        that volumes can be properly scaled.

        Parameters
        ----------
        memo : dict of str to [hydep.BurnableMaterial, int], optional
            Previously visited universes will populate this as they
            traverse the geometry. Needed for internal use, and modified
            through the traversal. Keys indicate ids of universes
            as they are discovered and will be updated.

        Returns
        -------
        Mapping[str, [hydep.BurnableMaterial, int]]
            Map of unique hashable IDs for unique burnable materials to
            the material and the number of instances. Should only contain
            information on this specific instance.
        """

    @abstractmethod
    def differentiateBurnableMaterials(self, memo=None):
        """Create new burnable materials and potentially mimic this universe

        This routine is important to create unique burnable materials
        that can be depleted using spatially correct fluxes and
        reaction rates.

        This method is expected to dig through contained universes and
        creates unique :class:`hydep.BurnedMaterial` objects and
        potentially new interior universes. This material itself may
        be cloned if the following conditions are met:

            1. At least one contained universe was cloned
               (e.g.a fuel pin was replaced)
            2. This object has been encountered before

        If at least one contained universe was cloned but this
        is the first time encountering this universe, the
        modifications will be made in-place, and the original
        returned.

        Parameters
        ----------
        memo : set of str, optional
            Set containing unique ids of previously found universes.
            Needed for internal use and not necessary for most end
            users.

        Returns
        -------
        Universe
            Either the originating universe or a near clone, but with
            one or more underlying materials changed.

        """

    def boundaries(self, memo=None):
        """Find boundaries in x, y, and z direction

        Also used to set :attr:`bounds`

        Parameters
        ----------
        memo : dict, optional
            Not usually required by end-user. Used to track which
            sub-universes have already been traversed. Will
            map universe ids to discovered boundaries

        Returns
        -------
        b : None or hydep.internal.Boundaries
            A value of ``None`` implies this object is unbounded
            in all directions. Otherwise, the ``x``, ``y``, and
            ``z`` attributes of the returned object describe
            the boundaries, potentially +/- ``numpy.inf``, for
            each dimension.

        """
        return None

    @property
    def bounds(self):
        if self._bounds is False:
            self.bounds = self.boundaries()
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

        if by is not None:
            assert all(isinstance(o, numbers.Real) for o in by)
            assert by[0] < by[1]

        if bz is not None:
            assert all(isinstance(o, numbers.Real) for o in bz)
            assert bz[0] < bz[1]

        self._bounds = Boundaries(bx, by, bz)

    @staticmethod
    def _compareUpdateBounds(current, new):
        # TODO Move off universe. Something shared by CartesianLattice, LatticeStack?
        if new[0] != -numpy.inf:
            if current[0] == -numpy.inf:
                mn = new[0]
            else:
                mn = min(current[0], new[0])
        else:
            mn = current[0]

        if new[1] != numpy.inf:
            if current[1] == numpy.inf:
                mx = new[1]
            else:
                mx = max(current[1], new[1])
        else:
            mx = current[1]
        return mn, mx


class InfiniteMaterial(Universe):
    """A physically unbounded space full of a single material

    Useful for filling a lattice position with a single material,
    e.g. moderator or structural material.

    Parameters
    ----------
    material : hydep.Material
        Material that occupies this eternal universe
    name : str, optional
        Name of this universe

    Attributes
    ----------
    material : hydep.Material
        Material that occupies this eternal universe
    name : str or None
        Name of this universe
    id : str
        Unique identifer for this universe
    """
    material = TypedAttr("material", Material)

    def __init__(self, material, name=None):
        self.material = material
        super().__init__(name)

    def findMaterials(self, memo=None):
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
        if memo is None or self.material.id not in memo:
            memo.add(self.material.id)
            yield self.material

    def countBurnableMaterials(self, _memo=None):
        """Count all occurances of burnable materials

        Useful prior to cloning new burnable materials, so
        that volumes can be properly scaled.

        Parameters
        ----------
        memo : dict of str to [hydep.BurnableMaterial, int], optional
            Previously visited universes will populate this as they
            traverse the geometry. Needed for internal use, and modified
            through the traversal. Keys indicate ids of universes
            as they are discovered and will be updated.

        Returns
        -------
        Mapping[str, [hydep.BurnableMaterial, int]]
            Map of unique hashable IDs for unique burnable materials to
            the material and the number of instances. Should only contain
            information on this specific instance.
        """
        if isinstance(self.material, BurnableMaterial):
            return {self.material.id: [self.material, 1]}
        return {}

    def differentiateBurnableMaterials(self, memo=None):
        """Create new burnable materials and potentially mimic this universe

        This routine is important to create unique burnable materials
        that can be depleted using spatially correct fluxes and
        reaction rates.

        This method digs through contained universes and creates unique
        :class:`hydep.BurnedMaterial` objects and potentially new
        interior universes. This material itself may be cloned if the
        following conditions are met:

            1. At least one contained universe was cloned
               (e.g.a fuel pin was replaced)
            2. This object has been encountered before

        If at least one contained universe was cloned but this
        is the first time encountering this universe, the
        modifications will be made in-place, and the original
        returned.

        Parameters
        ----------
        memo : set of str, optional
            Set containing unique ids of previously found universes.
            Needed for internal use and not necessary for most end
            users.

        Returns
        -------
        InfiniteMaterial
            Either the originating universe or a near clone, but with
            one or more underlying materials changed.

        """

        if isinstance(self.material, BurnableMaterial) or memo is not None:
            if self.material.id not in memo:
                memo.add(self.material.id)
                return self
            new = self.__class__(copy.deepcopy(self.material), self.name)
            memo.add(new.id)
            new.name += "_{}".format(new.id)
            return new
        return self
