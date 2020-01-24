"""
Classes for operating on fission yields

From OpenMC - MIT Licensed
Copyright: 2011-2019 Massachusetts Institute of Technology and
OpenMC collaborators

https://docs.openmc.org/en/latest/pythonapi/deplete.html

Changes made:
    1. FissionYield.products is tuple of int for ZZAAAI
    2. FissionYieldDistribution.products is a tuple of int
    3. FissionYieldDistribution.from_xml_element converts isotope
       names to ZAI identifiers
    4. Provided :meth:`FissionYieldDistribution.at`,
       :meth:`FissionYieldDistribution.get`,
       :meth:`FissionYieldDistribution.values`. Modified
       :meth:`FissionYieldDistribution.items` to use the
       ``at`` method.
"""

import bisect
from collections.abc import Mapping
from numbers import Real, Integral

from numpy import empty

from hydep.internal import getIsotope


__all__ = ["FissionYield", "FissionYieldDistribution"]


class FissionYieldDistribution(Mapping):
    """Energy-dependent fission product yields for a single nuclide

    Can be used as a dictionary mapping energies and products to fission
    yields::

        >>> fydist = FissionYieldDistribution(
        ...     {0.0253: {541350: 0.021}})
        >>> fydist[0.0253][541350]
        0.021

    Parameters
    ----------
    fission_yields : dict
        Dictionary of energies [eV] and fission product yields for that energy.
        Expected to be of the form ``{float: {str: float}}``. The first
        float is the energy, typically in eV, that represents this
        distribution. The underlying dictionary maps fission products
        to their respective yields.

    Attributes
    ----------
    energies : tuple of float
        Energies [eV] for which fission yields exist. Sorted by
        increasing energy
    products : tuple of int
        ZAI of fission products produced at all energies. Sorted by
        increasing value.
    yield_matrix : numpy.ndarray
        Array ``(n_energy, n_products)`` where
        ``yield_matrix[g, j]`` is the fission yield of product
        ``j`` for energy group ``g``.

    See Also
    --------
    * :class:`FissionYield` - Class used for storing yields at a given energy
    """

    def __init__(self, fission_yields):
        # mapping {energy: {product: value}}
        energies = sorted(fission_yields)

        # Get a consistent set of products to produce a matrix of yields
        shared_prod = set.union(*(set(x) for x in fission_yields.values()))
        ordered_prod = sorted(shared_prod)

        yield_matrix = empty((len(energies), len(shared_prod)))

        for g_index, energy in enumerate(energies):
            prod_map = fission_yields[energy]
            for prod_ix, product in enumerate(ordered_prod):
                yield_val = prod_map.get(product, 0.0)
                yield_matrix[g_index, prod_ix] = yield_val
        self.energies = tuple(energies)
        self.products = tuple(ordered_prod)
        self.yield_matrix = yield_matrix

    def __len__(self):
        return len(self.energies)

    def __getitem__(self, energy):
        if energy not in self.energies:
            raise KeyError(energy)
        return FissionYield(
            self.products, self.yield_matrix[self.energies.index(energy)]
        )

    def __iter__(self):
        return iter(self.energies)

    def __repr__(self):
        return "<{} with {} products at {} energies>".format(
            self.__class__.__name__, self.yield_matrix.shape[1], len(self.energies)
        )

    def at(self, index):
        """Return a set of fission yields at a given energy position

        Shortcut around not knowing a specific energy value, just
        a position. Most useful for grabbing the fission yields
        at the lowers (``index=0``) or highest (``index=-1``) energies

        Parameters
        ----------
        index : int
            Index in :attr:`energies` to pull fission yields from.
            index : int
                Index in :attr:`energies` to pull fission yields from.

        Returns
        -------
        FissionYield
            Fission yields evalutated at ``self.energies[index]``

        """
        if not isinstance(index, Integral):
            raise TypeError(
                "Only integer values allowed in {}.at, not {}".format(
                    type(self), type(index)))
        return FissionYield(self.products, self.yield_matrix[index])

    def get(self, energy, default=None, atol=1e-3):
        """Evaluate fission yields at an energy, with a fallback

        Act like :meth:`dict.get`, in that if ``energy`` does not
        exist, a default value can be returned. Allows for a little
        floating point tolerance

        Parameters
        ----------
        energy : float
            Desired energy that may or may not exist in :attr:`energies`.
        default : Optional[FissionYield]
            Default value to be returned if a set of yields at ``energy``
            could not be found. ``None`` is acceptable.
        atol : Optional[float]
            Absolute tolerance [eV] to apply to ``energy``. Defaults to
            1E-3 eV.

        Returns
        -------
        Union[FissionYield,default]
            If ``energy`` is found, return the :class:`FissionYield` for
            that specific energy. Otherwise, the default value is
            returned.

        """
        for ix, myenergy in enumerate(self.energies):
            if myenergy == energy or (-atol < myenergy - energy < atol):
                return self.at(ix)
        return default

    def values(self):
        """Iterate over all fission yields

        Yields
        ------
        FissionYield
            Fission yield for a specific energy

        """
        for row in self.yield_matrix:
            yield FissionYield(self.products, row)

    def items(self):
        """Iterate over pairs of energy and fission yields

        Yields
        ------
        float
            Energy [eV]
        FissionYield
            Fission yields at the provided energy
        """
        for ene, row in zip(self.energies, self.yield_matrix):
            yield ene, FissionYield(self.products, row)

    @classmethod
    def from_xml_element(cls, element):
        """Construct a distribution from a depletion chain xml file

        Parameters
        ----------
        element : xml.etree.ElementTree.Element
            XML element to pull fission yield data from

        Returns
        -------
        FissionYieldDistribution
        """
        all_yields = {}
        for yield_elem in element.iter("fission_yields"):
            energy = float(yield_elem.get("energy"))

            products = []
            for p in yield_elem.find("products").text.split():
                products.append(getIsotope(name=p).zai)

            yields = map(float, yield_elem.find("data").text.split())
            # Get a map of products to their corresponding yield
            all_yields[energy] = dict(zip(products, yields))

        return cls(all_yields)


class FissionYield(Mapping):
    """Mapping for fission yields for parent isotope.

    Can be used like a dictionary to fetch fission yields.
    Supports multiplication of a scalar to scale the fission
    yields and addition of another set of yields.

    Does not support resizing / inserting new products that do
    not exist.

    Parameters
    ----------
    products : tuple of int
        Sorted products for this specific distribution
    yields : numpy.ndarray
        Fission product yields for each product in ``products``

    Attributes
    ----------
    products : tuple of int
        Products for this specific distribution
    yields : numpy.ndarray
        Fission product yields for each product in ``products``

    Examples
    --------
    >>> import numpy
    >>> fy = FissionYield(
    ...     (531290, 541350, 621490),
    ...     numpy.array((0.001, 0.002, 0.0003)))
    >>> fy[541350]
    0.002
    >>> new = FissionYield(fy.products, fy.yields.copy())
    >>> fy *= 2
    >>> fy[541350]
    0.004
    >>> new[541350]
    0.002
    >>> (new + fy)[621490]
    0.0009
    >>> dict(new) == {541350: 0.002, 531290: 0.001, 621490: 0.0003}
    True
    """

    def __init__(self, products, yields):
        self.products = products
        self.yields = yields

    def __contains__(self, product):
        ix = bisect.bisect_left(self.products, product)
        return ix != len(self.products) and self.products[ix] == product

    def __getitem__(self, product):
        ix = bisect.bisect_left(self.products, product)
        if ix == len(self.products) or self.products[ix] != product:
            raise KeyError(product)
        return self.yields[ix]

    def __len__(self):
        return len(self.products)

    def __iter__(self):
        return iter(self.products)

    def items(self):
        """Return pairs of product, yield"""
        return zip(self.products, self.yields)

    def __add__(self, other):
        if not isinstance(other, FissionYield):
            return NotImplemented
        new = FissionYield(self.products, self.yields.copy())
        new += other
        return new

    def __iadd__(self, other):
        """Increment value from other fission yield"""
        if not isinstance(other, FissionYield):
            return NotImplemented
        self.yields += other.yields
        return self

    def __radd__(self, other):
        return self + other

    def __imul__(self, scalar):
        if not isinstance(scalar, Real):
            return NotImplemented
        self.yields *= scalar
        return self

    def __mul__(self, scalar):
        if not isinstance(scalar, Real):
            return NotImplemented
        new = FissionYield(self.products, self.yields.copy())
        new *= scalar
        return new

    def __rmul__(self, scalar):
        return self * scalar

    def __repr__(self):
        return "<{} containing {} products and yields>".format(
            self.__class__.__name__, len(self)
        )
