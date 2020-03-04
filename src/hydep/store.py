"""
Class for writting result data
"""

from abc import ABC, abstractmethod
import typing

__all__ = ["BaseStore"]


class BaseStore(ABC):
    """Abstract base class for storing data after transport and depletion

    Concrete classes are expected to provide three methods:

    .. autosummary::
        :toctree: generated

        beforeMain
        postTransport
        writeCompositions

    At construction, no data is expected to be written.

    Parameters
    ----------
    nCoarseSteps : int
        Number of coarse time steps used in the simulation. These
        values reflect the number of high fidelity transport
        simulations in the coupled sequence
    nTotalSteps : int
        Number of total time steps, including substeps, in the
        simulation. These correspond to be high fidelity and
        reduced order solutions
    nBurnableMaterials : int
        Number of burnable materials in the entire problem
    nGroups : int, optional
        Number of energy groups [default 1] for fluxes, cross
        sections, and reaction rates.

    Attributes
    ----------
    nCoarseSteps : int
        Number of coarse time steps used in the simulation. These
        values reflect the number of high fidelity transport
        simulations in the coupled sequence
    nTotalSteps : int
        Number of total time steps, including substeps, in the
        simulation. These correspond to be high fidelity and
        reduced order solutions
    nBurnableMaterials : int
        Number of burnable materials in the entire problem
    nGroups : int, optional
        Number of energy groups [default 1] for fluxes, cross
        sections, and reaction rates.
    VERSION : Tuple[int, int]
        Major and minor version of the stored data. Changes to major
        version will reflect new layouts and/or data has been removed.
        Changes to the minor version reflect new data has been added,
        or performance improvements. Scripts that work for ``x.y`` can
        be expected to also work for ``x.z``, but compatability between
        ``a.b`` and ``c.d`` is not guaranteed.

    """

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
        burnableIndexes : iterable of [int, str]
            Burnable material ids and names ordered how they
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

    @property
    @abstractmethod
    def VERSION(self) -> typing.Tuple[int, int]:
        pass
