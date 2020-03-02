"""
Base classes for transport solvers


Features
--------

:class:`HighFidelitySolver` and :class:`ReducedOrderSolver`
instances are coupled together based on the
:attr:`ReducedOrderSolver.needs` and :attr:`HighFidelitySolver.features`
sets. The two solvers are incompatible if
``rom.needs.difference(hf.features)`` is not a null set. Below
are the items that can be present in either set and what they entail,
both from the reduced order side and the high fidelity side.

``homog.macro.global`` : A code is capable of computing or needs
homogenized (potentially multi-group) macroscopic cross sections
over the entire domain.

``homog.macro.local`` : A code is capable of computing or needs
homogenized (potentially multi-group) macroscopic cross sections
over sub-domains, e.g. axial nodes in a fuel assembly, pin-level,
or sub-pin level.

``homog.micro.local`` : A code is capable of computing or needs
homogenized (potentially multi-group) microscopic cross sections
over sub-domains, e.g. axial nodes in a fuel assembly, pin-level,
or sub-pin level.

``fissionMatrix`` : A code is capable of computing or needs the
fission matrix in burnable materials

``reactionrates.local`` : A code is capable of computing or needs
local reaction rates
"""

from abc import ABC, abstractmethod

from hydep.internal import TransportResult
from hydep.internal.features import FeatureCollection
from hydep.exceptions import IncompatibityError


class TransportSolver(ABC):
    """Base class"""

    def beforeMain(self, model, manager, settings):
        """Execute any initial actions prior to main sequence

        By default this performs no actions, but can be used to
        create initial input files, initialize libraries, etc.

        Parameters
        ----------
        model : hydep.Model
            Geometry and materials
        manager : hydep.Manager
            Depletion interface containing information on time steps,
            powers, depletion chain, and substeps
        settings : hydep.settings.HydepSettings
            Settings for the entire interface

        """

    def finalize(self, success) -> None:
        """Perform any final actions before moving on or terminating

        This method will be called regardless if :meth:`execute`
        or :meth:`processResults` succeeded. Subclasses can take
        separate actions, or take no actions.

        Parameters
        ----------
        success : bool
            Flag indicating the success of the current solution
        """


class HighFidelitySolver(TransportSolver):
    """High fidelity transport solver"""

    # TODO Specify tempdir for execution?

    @property
    @abstractmethod
    def features(self) -> frozenset:
        """Features this solver is capable of

        Items should be subclass of :class:`hydep.features.Feature`
        """

    @abstractmethod
    def bosSolve(self, compositions, timestep, power) -> TransportResult:
        """Solve BOS transport and return results

        Relies upon :meth:`bosUpdate`, :meth:`execute`,
        :meth:`processResults` and :meth:`bosFinalize`

        Parameters
        ----------
        compositions : hydep.internal.CompBundle
            New compositions for this point in time such that
            ``compositions.densities[i][j]`` is the updated
            atom density for ``compositions.zai[j]`` for material
            ``i``
        timestep : hydep.internal.TimeStep
            Current point in calendar time for the beginning
            of this coarse step
        power : float
            Current power [W]

        Returns
        -------
        TransportResult
            Flux, multiplication factor, and run time. Run time
            is computed from :meth:`bosExecute`. Additional data
            should be included corresponding to hooks
        """

    def eolSolve(self, compositions, timestep, power) -> TransportResult:
        """Perform the final transport solution

        No further transport solutions will follow. If not overwritten,
        defer back to :meth:`bosSolve`. Provided for subclasses that
        wish to change how the final step is handled.

        Parameters
        ----------
        compositions : hydep.internal.CompBundle
            New compositions for this point in time such that
            ``compositions.densities[i][j]`` is the updated
            atom density for ``compositions.zai[j]`` for material
            ``i``
        timestep : hydep.internal.TimeStep
            Current point in calendar time for the beginning
            of this coarse step
        power : float
            Current power [W]

        Returns
        -------
        TransportResult
            Flux, multiplication factor, and run time. Run time
            is computed from :meth:`bosExecute`. Additional data
            should be included corresponding to hooks

        """
        return self.bosSolve(compositions, timestep, power)

    @abstractmethod
    def setHooks(self, needs):
        """Set any pre and post-run hooks

        Hooks should provide sufficient information that the
        reduced order code can be executed, e.g. getting spatially
        homogenized cross section, building the fission matrix, etc.
        Hooks should be executed in either :meth:`bosUpdate` or
        :meth:`beforeMain`. Information generated here should
        inform the post process routine :meth:`process` to add
        necessary information to the
        :class:`hydep.internal.TransportResult`

        Parameters
        ----------
        needs : hydep.internal.feature.FeatureCollection
            The needs of both the reduced order solver
            and the depletion manager

        """

    def checkCompatibility(self, solver):
        """Ensure coupling between solvers and/or depletion manager

        Parameters
        ----------
        solver : ReducedOrderSolver or hydep.Manager
            Specific reduced order solver coupled to this high
            fidelity solver. Will have a ``needs`` attribute
            that describes the various features needed in order
            to successfully couple the two solvers.

        Raises
        ------
        hydep.IncompatibityError
            If the reduced order solver needs features this
            solver does not have

        """

        if not solver.needs.issubset(self.features):
            difference = solver.needs.difference(self.features)
            tails = []
            if difference.features:
                names = sorted(f.name for f in difference.features)
                tails.append(f'Features: {", ".join(names)}')
            if difference.macroXS:
                tails.append(
                    'Macroscopic cross sections: '
                    f'{", ".join(sorted(difference.macroXS))}'
                )
            msg = "\n".join(tails)
            raise IncompatibityError(
                f"Cannot couple {self.__class__} to {solver.__class__}."
                f"\nDifference in features: {msg}"
            )


class ReducedOrderSolver(TransportSolver):
    """Reduced order solver"""

    @property
    @abstractmethod
    def needs(self) -> FeatureCollection:
        """Features needed by this reduced order solver"""

    @abstractmethod
    def substepSolve(
        self, timestep, compositions, microCrossSections
    ) -> TransportResult:
        """Solve reduced order problem at a substep

        Parameters
        ----------
        timestep : hydep.internal.TimeStep
            Current time step information
        compositions : hydep.internal.CompBundle
            Updated compositions for this ``timestep``
        microCrossSections : iterable of hydep.internal.MicroXsVector
            Microscopic cross sections extrapolated for this time step.
            Each entry corresponds to a burnable material, ordered
            consistent with the remainder of the framework

        Returns
        -------
        TransportResult
            At least flux, keff, and run time for this substep solution

        """

    def processBOS(self, txResult, timestep, power):
        """Process data from the beginning of step high fidelity solution

        Not required to be implemented, but some solvers may want to
        store initial values.

        Parameters
        ----------
        txResult : hydep.internal.TransportResult
            Transport result from the :class:`HighFidelitySolver`
        timestep : hydep.internal.TimeStep
            Representation of the point in calendar time, and
            step in simulation.
        power : float
            Current system power[W]. To be held constant across the
            substep interval.

        """
