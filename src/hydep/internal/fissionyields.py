"""
Class for operating on fission yields

From OpenMC - MIT Licensed
Copyright: 2011-2019 Massachusetts Institute of Technology and
OpenMC collaborators

https://docs.openmc.org/en/latest/pythonapi/deplete.html

Changes made:
    1. FissionYield.products is tuple of int for ZZAAAI
"""

import bisect
from collections.abc import Mapping
from numbers import Real

__all__ = ["FissionYield"]


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
