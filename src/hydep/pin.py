"""Cylindrical pin implementation"""
from collections.abc import Iterable
import numbers

import numpy

from hydep.lib import Universe
from hydep import Material, BurnableMaterial
from .typed import IterableOf, TypedAttr


__all__ = ("Pin", )


class Pin(Universe):
    """Infinite concentric cylinders of materials

    Can be iterated over and indexed, but not with
    slices [results not guarunteed]. Psuedo-mutable,
    as :attr:`radii` and :attr:`materials` cannot be set
    using indexing, but can be over-written.

    Parameters
    ----------
    radii : Iterable[float]
        Outer radii of each material. Must be positive and
        increasing.
    materials : Iterable[hydep.Material]
        Materials to fill each section defined in :attr:`radii`.
        Must have identical length to ``radii``
    outer : hydep.Material
        Material to extend from the last radius out to infinity.
    name : str, optional
        Name of this pin

    Attributes
    ----------
    radii : tuple of float
        Outer radii of each material
    materials : tuple of hydep.Material
        Materials to fill each section defined in :attr:`radii`.
    outer : hydep.Material
        Material to extend from the last radius out to infinity.
    name : str or None
        Name of this pin

    """
    outer = TypedAttr("outer", Material)
    radii = IterableOf("radii", numbers.Real)
    materials = IterableOf("materials", Material)

    def __init__(self, radii, materials, outer, name=None):
        assert isinstance(radii, Iterable)
        assert isinstance(materials, Iterable)
        assert len(radii) == len(materials)

        prev = 0
        for ix, r in enumerate(radii):
            assert r > prev, (ix, r)
            prev = r

        super().__init__(name)
        self.outer = outer
        self.radii = tuple(radii)
        self.materials = tuple(materials)

    def __iter__(self):
        for r, m in zip(self.radii, self.materials):
            yield r, m
        yield numpy.inf, self.outer

    def __len__(self):
        return len(self.radii) + 1

    def __getitem__(self, index):
        # TODO Support slicing
        if isinstance(index, numbers.Integral):
            if index < 0:
                index = len(self) + index
            if index == len(self):
                return numpy.inf, self.outer
        else:
            raise TypeError(index)
        return self.radii[index], self.materials[index]

    def findMaterials(self, memo=None):
        """Yield all materials present.

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
        memo = set() if memo is None else memo
        for _r, mat in self:
            hid = id(mat)
            if hid in memo:
                continue
            memo.add(hid)
            yield mat

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
        local = {}
        for _r, mat in self:
            if not isinstance(mat, BurnableMaterial):
                continue
            hid = mat.id
            repeat = local.get(hid)
            if repeat is None:
                local[hid] = [mat, 1]
            else:
                repeat[1] += 1
        return local

    def differentiateBurnableMaterials(self, memo=None):
        """Create new burnable materials and potentially mimic this pin

        This routine is important to create unique burnable materials
        that can be depleted using spatially correct fluxes and
        reaction rates.

        This method digs through contained materials and creates unique
        :class:`hydep.BurnedMaterial` objects.This material itself may
        be cloned if the following conditions are met:

            1. At least one contained material was cloned
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
        Pin
            Either the originating universe or a near clone, but with
            one or more underlying materials changed.

        """
        # TODO Explain ^^^
        memo = set() if memo is None else memo
        updates = {}
        for index, (_r, mat) in enumerate(self):
            if not isinstance(mat, BurnableMaterial):
                continue
            if id(mat) in memo:
                mat = updates[index] = mat.copy()
                # TODO remove assumption of always hex ids
                mat.name = "{}_copy{}".format(mat.name.split("_copy")[0], mat.id)
            memo.add(id(mat))

        if not updates:
            memo.add(id(self))
            return self

        outer = updates.pop(len(self.materials), self.outer)
        materials = [updates.get(ix, mat) for ix, mat in enumerate(self.materials)]

        if id(self) not in memo:
            memo.add(id(self))
            self.outer = outer
            self.materials = materials
            return self

        new = self.__class__(self.radii, materials, outer)
        if self.name is not None:
            new.name = "{}_copy{}".format(self.name.split("_copy")[0], new.id)
        return new
