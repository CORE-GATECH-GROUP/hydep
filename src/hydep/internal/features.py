"""Basic feature management system

Used to indicate what features / physics are needed to
couple the reduced order code to the high fidelity code
"""

from collections import namedtuple
from collections.abc import Collection
from itertools import chain


Feature = namedtuple("Feature", ["name", "description"])


FISSION_MATRIX = Feature("fission matrix", "the fission matrix")
HOMOG_GLOBAL = Feature(
    "global homogenization",
    "homogenized macroscopic group constants across the entire domain",
)
HOMOG_LOCAL = Feature(
    "local homogenization",
    "homogenized macroscopic group constants across arbitrary sub-domain",
)
MICRO_REACTION_XS = Feature(
    "microscopic cross sections",
    "microscopic reaction cross sections across arbitrary sub-domain",
)
FISSION_YIELDS = Feature(
    "fission yields",
    "spectrum-representative fission yields for nuclides in burnable "
    "regions",
)


class FeatureCollection(Collection):
    """Set-like collection of features

    Intended to be placed on a solver to describe the capabilities
    of a high-fidelity code or needs of a reduced order solver.

    Implements a ``Collection`` interface by providing
    :meth:`__len__`, :meth:`__iter__`, and :meth:`__contains__`
    methods. Set-like, in that it can be compared to another
    instance using :meth:`__eq__`, or :meth:`issubset`, and
    supports creating a :meth:`union` of another collection.

    Parameters
    ----------
    features : Iterable[Feature], optional
        Collection of features provided (or needed) by a specific
        solver
    macroXS : Iterable[str], optional
        Collection of homogenized macroscopic cross sections
        provided (or needed) by a specific solver

    Attributes
    ----------
    features : frozenset of Feature
        Collection of features provided (or needed) by a specific
        solver. Read-only, and allowed to be empty
    macroXS : frozenset of str
        Collection of homogenized macroscopic cross sections
        provided (or needed) by a specific solver. Read-only,
        and allowed to be empty

    """

    __slots__ = ("_features", "_xs")

    def __init__(self, features=None, macroXS=None):
        if features is None or not features:
            self._features = frozenset()
        else:
            for item in features:
                if not isinstance(item, Feature):
                    raise TypeError(
                        "Found {} in features when should be {}".format(
                            type(item), Feature))
            self._features = frozenset(features)

        if macroXS is None or not macroXS:
            self._xs = frozenset()
        else:
            self._xs = frozenset(macroXS)

    @property
    def features(self) -> frozenset:
        return self._features

    @property
    def macroXS(self) -> frozenset:
        return self._xs

    def __len__(self) -> int:
        """Total number of items stored"""
        return len(self.macroXS) + len(self.features)

    def __contains__(self, key) -> bool:
        """Search for a specific item in the collection"""
        return key in self.features or key in self.macroXS

    def __iter__(self):
        """Iterate first over features, then macroXS"""
        return chain(self.features, self.macroXS)

    def __eq__(self, other) -> bool:
        """Test equality of two :class:`FeatureCollection`"""
        if isinstance(other, self.__class__):
            return self.features == other.features and self.macroXS == other.macroXS
        return NotImplemented

    def __hash__(self):
        return hash(self.features.union(self.macroXS))

    def union(self, other):
        """Create an new collection with the contents of another

        Parameters
        ----------
        other : FeatureCollection
            Object of the union

        Returns
        -------
        FeatureCollection
            A new collection that contains all the :attr:`features`
            and :attr:`macroXS` of this and ``other``

        Examples
        --------
        >>> s = FeatureCollection({FISSION_MATRIX, })
        >>> o = FeatureCollection({MICRO_REACTION_XS, HOMOG_GLOBAL}, {"INF_ABS"})
        >>> u = s.union(o)
        >>> all(f in u.features for f in s.features)
        True
        >>> all(f in u.features for f in o.features)
        True
        >>> all(f in u.macroXS for f in s.macroXS)
        True
        >>> all(f in u.macroXS for f in o.macroXS)
        True

        """
        if not isinstance(other, self.__class__):
            raise TypeError("Expected other to be {}, got {}".format(
                self.__class__, type(other)))
        return self.__class__(self.features.union(other.features),
                              self.macroXS.union(other.macroXS))

    def __bool__(self):
        return len(self) > 0

    def issubset(self, other) -> bool:
        """Return true if every element is contained in other

        Parameters
        ----------
        other : FeatureCollection
            Class that may or may not contain all the features
            and cross section on this instance

        Returns
        -------
        bool
            True if every element of :attr:`features` and
            :attr:`macroXS` is contained in ``other``

        Examples
        --------
        >>> s = FeatureCollection({FISSION_MATRIX, })
        >>> o = FeatureCollection({MICRO_REACTION_XS, HOMOG_GLOBAL}, {"INF_ABS"})
        >>> u = s.union(o)
        >>> s.issubset(o)
        False
        >>> s.issubset(u)
        True
        >>> o.issubset(u)
        True

        """
        if not isinstance(other, self.__class__):
            raise TypeError("Expected other to be {}, got {}".format(
                self.__class__, type(other)))
        return (self.features.issubset(other.features)
                and self.macroXS.issubset(other.macroXS))

    # TODO Maybe provide __getattr__ to cover all frozen-set features
    def difference(self, other):
        """Return a new collection with elements that are not in other

        Parameters
        ----------
        other : FeatureCollection
            Alternative features

        Returns
        -------
        FeatureCollection
            Collection of features that are present here, but not
            in ``other``

        Examples
        --------
        >>> s = FeatureCollection({FISSION_MATRIX, })
        >>> o = FeatureCollection({MICRO_REACTION_XS, HOMOG_GLOBAL}, {"INF_ABS"})
        >>> d = s.difference(o)
        >>> d.features == frozenset({FISSION_MATRIX,})
        True
        >>> d.macroXS == frozenset()
        True

        """
        if not isinstance(other, self.__class__):
            raise TypeError(type(other))
        return self.__class__(
            self.features.difference(other.features),
            self.macroXS.difference(other.macroXS),
        )
