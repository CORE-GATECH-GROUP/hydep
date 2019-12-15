import numbers
from collections.abc import Iterable, Generator

import numpy


from hydep import Universe, Pin, InfiniteMaterial, Material
from .typed import BoundedTyped, TypedAttr
from hydep.internal import Boundaries

__all__ = ("LatticeStack",)


class LatticeStack(Universe):
    """Representation of a 1D stack of universes

    Coded up to reflect a vertical stack with the :attr:`nLayers`
    and :attr:`heights` attributes. Providing the :attr:`items`,
    :attr:`heights`, and :attr:`outer` attributes is likely
    required to fully represent this stack.

    Fulfills a :class:`~collections.abc.Sequence` interface, wherein
    the stack can be iterated over, indexed, and each index can be set
    directly through the object.

    Parameters
    ----------
    nLayers : int
        Positive integer for the number of layers
    heights : Iterable[float], optional
        Boundaries for each layer, so must have :attr:`nLayers` + 1 entries. Must
        all be increasing
    items : Iterable[hydep.Universe], optional
        Iterable of universes that fill each layer.
        Must have :attr:`nLayers` elements.
        Ordered such that ``items[i]`` occupies the space between ``heights
    name : str, optional
        Name of this stack
    outer : hydep.Material, optional
        Material that fills the space outside this stack

    Attributes
    ----------
    nLayers : int
        Number of layers
    heights : Iterable[float], optional
        Boundaries for each layer. Must have :attr:`nLayers` + 1 entries.
        Must all be increasing in value.
    items : Iterable[hydep.Universe], optional
        Iterable of universes that fill each layer. Must have
        :attr:`nLayers` elements. Ordered such that ``items[i]`` occupies
        the space between ``heights[i]`` and ``heights[i + 1]``
    name : str or None
        Name of this stack
    outer : hydep.Material or None
        Material that fills the space outside this stack
    id : str
        Unique identifier
    bounds : None or Iterable[Iterable[float]] or :class:`hydep.internal.Boundaries`
        Spatial bounds for this universe. A value of ``None`` implies
        unbounded in space.

    Examples
    --------
    >>> import hydep
    >>> m = hydep.Material("material")
    >>> u = hydep.InfiniteMaterial(m, name="empty" )
    >>> stack = hydep.LatticeStack(2, [0, 0.5, 1], [u, u])
    >>> stack[0]
    <InfiniteMaterial empty at ...>
    >>> stack[0] is u
    True
    >>> for ix, item in enumerate(stack):
    ...     assert item is stack[ix]

    """

    _nLayers = BoundedTyped("nLayers", numbers.Integral, gt=0)
    outer = TypedAttr("outer", Material, allowNone=True)

    def __init__(self, nLayers, heights=None, items=None, name=None, outer=None):
        super().__init__(name)
        self._nLayers = nLayers
        self._items = None
        self._heights = None
        if heights is not None:
            self.heights = heights
        if items is not None:
            self.items = items
        self.outer = outer

    def __len__(self):
        """Number of layers in this stack"""
        return self._nLayers

    def __iter__(self):
        """Iterate over all :attr:`items`"""
        return iter([] if self._items is None else self._items)

    def __getitem__(self, index) -> Universe:
        """Return the :class:`hydep.Universe` at ``index``"""
        if self._items is None:
            raise AttributeError("Vertical stack not set")
        return self._items[index]

    def __setitem__(self, index, universe):
        """Set the :class:`hydep.Universe` at ``index`` to be ``universe``"""
        if self._items is None:
            raise AttributeError("Vertical stack not set")
        assert isinstance(index, numbers.Integral)
        assert isinstance(universe, Universe)
        self._items[index] = universe

    def allocate(self):
        """Create :attr:`items` array but do not populate"""
        if self._items is not None:
            return
        self._items = numpy.empty(self.nLayers, dtype=object)

    @property
    def nLayers(self):
        return self._nLayers

    @property
    def heights(self):
        return self._heights

    @heights.setter
    def heights(self, heights):
        heights = numpy.asarray(heights, dtype=float).reshape(self.nLayers + 1)
        prev = heights.min()
        assert heights[0] == prev
        for ix, value in enumerate(heights[1:], start=1):
            assert value > prev
            prev = value
        self._heights = heights

    @property
    def items(self):
        return self._items

    @items.setter
    def items(self, items):
        if self.items is None:
            self.allocate()
        elif items is None:
            self._items = None

        assert isinstance(items, Iterable)
        assert len(items) == self.nLayers
        assert all(isinstance(x, Universe) for x in items)

        for ix, item in enumerate(items):
            self._items[ix] = item

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
        if self._items is None:
            raise AttributeError("Vertical stack {} not set".format(self))
        memo = set() if memo is None else memo
        for item in self.items:
            hid = item.id
            if hid in memo:
                continue
            for m in item.findMaterials(memo):
                yield m
            memo.add(hid)
        if self.outer is not None and self.outer.id not in memo:
            memo.add(self.outer.id)
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
            the material and the number of instance found on this instance.
        """
        # TODO Share with CartesianLattice?
        if self.items is None:
            raise AttributeError("Vertical stack {} not set".format(self))
        memo = {} if memo is None else memo
        local = {}
        for item in self.items:
            hid = item.id
            previous = memo.get(hid)
            if previous is None:
                memo[hid] = previous = item.countBurnableMaterials(memo)
            for uid, (mat, count) in previous.items():
                present = local.get(uid)
                if present is None:
                    local[uid] = [mat, count]
                else:
                    local[uid][1] += count
        return local

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
        LatticeStack
            Either the originating universe or a near clone, but with
            one or more underlying materials changed.

        """
        # TODO Support for outer - assume not burnable
        if self.items is None:
            raise AttributeError("Vertical stack not set for {}".format(self))
        memo = set() if memo is None else memo
        updates = {}

        for ix, item in enumerate(self):
            new = item.differentiateBurnableMaterials(memo)
            if new is not item:
                updates[ix] = new
            memo.add(new.id)

        if not updates:
            memo.add(self.id)
            return self

        newitems = numpy.empty_like(self.items)
        for ix, item in enumerate(self):
            newitems[ix] = updates.get(ix, item)

        if self.id not in memo:
            memo.add(self.id)
            self.items = newitems
            return self

        return self.__class__(self.nLayers, self.heights, newitems, self.name, self.outer)

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
        memo = {} if memo is None else memo

        if self._bounds is not False:
            memo[self.id] = self._bounds
            return self._bounds

        bx = (-numpy.inf, numpy.inf)
        by = (-numpy.inf, numpy.inf)

        for u in self.items:
            if isinstance(u, (Pin, InfiniteMaterial)):
                continue
            bounds = memo.get(u.id, False)  # None is a valid answer
            if bounds is False:
                memo[u.id] = bounds = u.boundaries(memo)

            if bounds is None:
                continue

            if bounds.x is not None:
                bx = self._compareUpdateBounds(bx, bounds.x)

            if bounds.y is not None:
                by = self._compareUpdateBounds(by, bounds.y)

        if -bx[0] == bx[1] == numpy.inf:
            bx = None
        if -by[0] == by[1] == numpy.inf:
            by = None

        mybounds = Boundaries(bx, by, (self.heights[0], self.heights[-1]))
        memo[self.id] = mybounds
        return mybounds
