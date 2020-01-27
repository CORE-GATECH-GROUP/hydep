"""Cartesian lattice implementation"""
import numbers
from collections.abc import Iterable

import numpy

from hydep.lib import Universe
from hydep import Pin, InfiniteMaterial, Material, BurnableMaterial
from .typed import BoundedTyped, TypedAttr
from hydep.internal import Boundaries


__all__ = ("CartesianLattice",)


class CartesianLattice(Universe):
    """A regular Cartesian lattice of universes

    Setting :attr:`array` is necessary to ensure full functionaltiy.
    Setting :attr:`outer` may also be necessary, especially if this
    object is placed inside another.

    Parameters
    ----------
    nx : int
        Number of items in the x direction
    ny : int
        Number of items in the y direction
    pitch : float
        Center to center distance between adjacent items
    array : iterable of iterable of hydep.lib.Universe optional
        Initial array of universes.
    name : str, optional
        Name of this lattice. No need for uniquness
    outer : hydep.Material, optional
        Material that resides outside this lattice

    Attributes
    ----------
    nx : int
        Number of items in the x direction
    ny : int
        Number of items in the y direction
    pitch : float
        Center to center distance between adjacent items
    array : itererable of iterable of hydep.lib.Universe or None
        Initial array of universes.
    name : str or None
        Name of this lattice. No need for uniquness
    outer : hydep.Material or None
        Material that resides outside this lattice
    bounds : None or Iterable[Iterable[float]] or hydep.internal.Boundaries
        Spatial bounds for this universe. A value of ``None`` indicates
        this universe is unbounded in all directions. Otherwise,
        a :class:`~hydep.internal.Boundaries` instance will be
        used to describe the x, y, and z boundaries.
    size : int
        Number of elements is the array
    shape : Iterable[int]
        Number of y and x elements
    flat : numpy.flatiter
        Iterator / accessor / setter that allows for single
        integer indexing.

    Examples
    --------
    >>> import hydep
    >>> m = hydep.Material("hydrogen", adens=1, H1=1)
    >>> u = hydep.InfiniteMaterial(m, name="inf H1")
    >>> lattice = hydep.CartesianLattice(2, 2, 1.2, [[u, u], [u, u]])
    >>> lattice[0, 0]
    <InfiniteMaterial inf H1 ...>
    >>> lattice.shape
    (2, 2)
    >>> lattice.size
    4
    >>> lattice.flat[3] is lattice[1, 0]
    True
    >>> for item in lattice.flat:
    ...     assert item is u

    """

    pitch = BoundedTyped("pitch", numbers.Real, gt=0.0)
    _nx = BoundedTyped("nx", numbers.Integral, gt=0)
    _ny = BoundedTyped("ny", numbers.Integral, gt=0)
    outer = TypedAttr("outer", Material, allowNone=True)

    def __init__(self, nx, ny, pitch, array=None, name=None, outer=None):
        super().__init__(name)
        self._nx = nx
        self._ny = ny
        self.pitch = pitch
        self._array = None
        if array is not None:
            self.array = array
        self.outer = outer

    @classmethod
    def fromMask(cls, pitch, maskarray, fillmap, **kwargs):
        """Create a new lattice using a mask or template

        Parameters
        ----------
        pitch : float
            Center to center distance between adjacent items
        maskarray : Iterable[Iterable[int]]
            Template for this array where each item is a key
            in ``fillmap``. Every location in the resulting
            array will be filled according to this template
        fillmap : Mapping[int, hydep.lib.Universe]
            Map used to populate final array
        kwargs:
            Additional arguments to be passed to the constructor

        Returns
        -------
        CartesianLattice
            Lattice with a populated :attr:`array` corresponding
            to the template. Shapes will be identical

        Examples
        --------
        >>> import hydep
        >>> m = hydep.Material("hydrogen", adens=1, H1=1)
        >>> u = hydep.InfiniteMaterial(m, name="inf H1")
        >>> o = hydep.InfiniteMaterial(m, name="other H1")
        >>> lookup = {0: u, 1: o}
        >>> mask = [[0, 1], [1, 0]]
        >>> lattice = hydep.CartesianLattice.fromMask(
        ...        1.2, mask, lookup)
        >>> lattice.shape
        (2, 2)
        >>> lattice[0, 0]
        <InfiniteMaterial inf H1 ...>
        >>> lattice[0, 1] is lookup[mask[0][1]]
        True
        """
        mask = numpy.asarray(maskarray, dtype=int)
        assert len(mask.shape) == 2
        assert not set(numpy.unique(mask)).difference(fillmap.keys())
        uarray = numpy.empty_like(mask, dtype=object)

        for key, value in fillmap.items():
            assert isinstance(value, Universe)
            # TODO resolve __array__ dispatching?
            # uarray[mask == key] = value
            for r, c in zip(*numpy.where(mask == key)):
                uarray[r, c] = value

        return cls(uarray.shape[1], uarray.shape[0], pitch, uarray, **kwargs)

    def __getitem__(self, pos):
        if self.array is None:
            raise AttributeError("Array not set")
        return self._array[pos]

    def __setitem__(self, rowcol, universe):
        if self.array is None:
            self.allocate()
        assert isinstance(rowcol, Iterable)
        assert len(rowcol) == 2
        assert all(isinstance(x, numbers.Integral) for x in rowcol)
        assert isinstance(universe, Universe)
        self._array[rowcol] = universe

    def __iter__(self):
        return iter(self._array if self._array is not None else [])

    def __len__(self):
        return self._ny

    @property
    def nx(self):
        return self._nx

    @property
    def ny(self):
        return self._ny

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, array):
        if array is None:
            self._array = None
            return

        assert len(array) == len(self)
        items = []
        for rowx, item in enumerate(array):
            assert len(item) == self._nx, (rowx, type(item), self._nx)
            for colx, u in enumerate(item):
                assert isinstance(u, Universe), (rowx, colx, type(u))
                items.append((rowx, colx, u))
        assert len(items) == self._ny * self._nx, (len(items), self._ny, self._nx)

        if self._array is None:
            self.allocate()

        for rowx, colx, u in items:
            self._array[rowx, colx] = u

    @property
    def size(self):
        return self._nx * self._ny

    @property
    def shape(self):
        return (self._ny, self._nx)

    @property
    def flat(self):
        if self._array is None:
            raise AttributeError("Array not set for {}".format(self))
        return self._array.flat

    def allocate(self):
        """Create and overwrite the existing array"""
        self._array = numpy.empty(self.shape, dtype=object)

    def findMaterials(self, memo=None):
        """Yield all materials present in this and contained universes

        Parameters
        ----------
        memo : set, optional
            Set containing ids of previously visited materials. Don't
            pass unless you know what you're doing. If given, will
            be modified with discovered materials

        Yields
        ------
        hydep.Material
            The first occurance of this material.

        """
        if self.array is None:
            raise AttributeError("Array not set for {}".format(self))
        memo = set() if memo is None else memo
        for item in self.flat:
            hid = id(item)
            if hid in memo:
                continue
            for m in item.findMaterials(memo):
                yield m
            memo.add(hid)
        if self.outer is not None and id(self.outer) not in memo:
            memo.add(id(self.outer))
            yield self.outer

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
        if self.array is None:
            raise AttributeError("Array not set for {}".format(self))
        memo = {} if memo is None else memo
        local = {}
        for u in self.flat:
            hid = id(u)
            previous = memo.get(hid)
            if previous is None:
                memo[hid] = previous = u.countBurnableMaterials(memo)
            for uid, (mat, count) in previous.items():
                # TODO Use local.setdefault?
                present = local.get(uid)
                if present is None:
                    local[uid] = [mat, count]
                else:
                    local[uid][1] += count

        if isinstance(self.outer, BurnableMaterial):
            oid = id(self.outer)
            present = local.get(oid)
            if present is None:
                local[oid] = [self.outer, 1]
            else:
                local[oid][1] += 1

        return local

    def differentiateBurnableMaterials(self, memo=None):
        """Create new burnable materials and potentially mimic this lattice

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
        CartesianLattice
            Either the originating universe or a near clone, but with
            one or more underlying materials changed.

        See Also
        --------
        hydep.Pin.differentialBurnableMaterials

        """
        # TODO Treatment for outer - assume not burnable
        if self.array is None:
            raise AttributeError("Array not set for {}".format(self))
        memo = set() if memo is None else memo
        updates = {}

        for ix, item in enumerate(self.flat):
            new = item.differentiateBurnableMaterials(memo)
            if new is not item:
                updates[ix] = new
            memo.add(id(new))

        if not updates:
            memo.add(id(self))
            return self

        newarray = numpy.empty_like(self.array)
        for ix, orig in enumerate(self.flat):
            newarray.flat[ix] = updates.get(ix, orig)

        if id(self) not in memo:
            memo.add(id(self))
            self.array = newarray
            return self

        return self.__class__(
            self.nx, self.ny, self.pitch, array=newarray, name=self.name,
        )

    def boundaries(self, memo=None):
        memo = {} if memo is None else memo

        if self._bounds is not False:
            memo[id(self)] = self._bounds
            return self._bounds

        bz = (-numpy.inf, numpy.inf)

        for u in self.flat:
            if isinstance(u, (Pin, InfiniteMaterial)):
                continue
            bounds = memo.get(u.id, False)  # None is a valid answer
            if bounds is False:
                memo[u.id] = bounds = u.boundaries(memo)
            if bounds is None:
                continue
            if bounds.z is not None:
                bz = self._compareUpdateBounds(bz, bounds.z)

        if bz[0] == -numpy.inf and bz[1] == numpy.inf:
            bz = None

        bx = (-self.nx * self.pitch * 0.5, self.nx * self.pitch * 0.5)
        by = (-self.ny * self.pitch * 0.5, self.ny * self.pitch * 0.5)

        mybounds = Boundaries(bx, by, bz)
        memo[id(self)] = mybounds
        return mybounds
