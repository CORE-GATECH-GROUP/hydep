"""
Class for writting result data
"""

from abc import ABC, abstractmethod
import typing

__all__ = ["BaseStore"]


class BaseStore(ABC):
    """Abstract base class for storing data after transport and depletion"""

    @abstractmethod
    def beforeMain(self, nhf, ntransport, ngroups, isotopes, burnableIndexes) -> None:
        """Called before main simulation sequence

        Parameters
        ----------
        nhf : int
            Number of high-fidelity transport solutions that will be
            used in this sequence. Also equal to one more than the
            number of coarse depletion intervals
        ntransport : int
            Number of total transport solutions (high fidelity and
            reduced oder) employed in this sequence. Also equal to
            one more than the number of depletion events (including
            substeps)
        ngroups : int
            Number of energy groups values like fluxes and cross sections
            will contain
        isotopes : tuple of hydep.internal.Isotope
            Isotopes used in the depletion chain
        burnableIndexes : iterable of [int, str, float]
            Each item is a 3-tuple of material id, name, and volume.
            Entries are ordered consistent to how the material are ordered
            are used across the sequence

        """

    @abstractmethod
    def postTransport(self, timeStep, transportResult) -> None:
        """Store transport results

        Transport results will come both after high fidelity
        and reduced order solutions.

        Parameters
        ----------
        timeStep : hydep.internal.TimeStep
            Point in calendar time from where these results were
            generated
        transportResult : hydep.internal.TransportResult
            Collection of data. Guaranteed to have at least
            a ``flux`` and ``keff`` attribute that are not
            ``None``

        """

    @abstractmethod
    def writeCompositions(self, timeStep, compBundle) -> None:
        """Write compositions for a given point in time

        Parameters
        ----------
        timeStep : hydep.internal.TimeStep
            Point in calendar time that corresponds to the
            compositions, e.g. compositions are from this point
            in time
        compBundle : hydep.internal.CompBundle
            New compositions. Will contain ordering of isotopes and
            compositions ordered consistent with the remainder
            of the sequence and corresponding argument to
            :meth:`beforeMain`

        """
