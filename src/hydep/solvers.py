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

    def beforeMain(self, model, orderedBumats):
        """Execute any initial actions prior to main sequence

        By default this performs no actions, but can be used to
        create initial input files, initialize libraries,
        etc.

        Parameters
        ----------
        model : hydep.Model
            Geometry and materials
        orderedBumats : tuple of hydep.BurnableMaterials
            Burnable materials ordered how the depletion system anticipates

        """

    @abstractmethod
    def execute(self) -> float:
        """Execute the simulation

        Returns
        -------
        float
            Wall time [s] required for simulation

        Raises
        ------
        hydep.FailedSolverError
            This exception, or an appropriate subclass, should
            be raised to indicate some fatal error occured
            in the solution
        """

    @abstractmethod
    def processResults(self) -> TransportResult:
        """Process output of solution

        Returns
        -------
        TransportResult
            Containing at least the flux and multiplication
            factor.

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

    def _solve(self) -> TransportResult:
        success = True
        try:
            runTime = self.execute()
            res = self.processResults()
            res.runTime = runTime
        except Exception as ee:
            success = False
            raise ee
        finally:
            self.finalize(success)

        return res

    def configure(self, config):
        """Configure the solver

        Parameters
        ----------
        config : :class:`configparser.ConfigParser`
            Configuration object that has loaded in user settings,
            either from a file, dictionary, or in memory.
        """


class HighFidelitySolver(TransportSolver):
    """High fidelity transport solver"""

    # TODO Some level of logging at each method
    # TODO Verbosity control for solution
    # TODO Specify tempdir for execution?

    @property
    @abstractmethod
    def features(self) -> frozenset:
        """Features this solver is capable of

        Items should be subclass of :class:`hydep.features.Feature`
        """

    @abstractmethod
    def processResults(self) -> TransportResult:
        """Process output of solution

        Returns
        -------
        TransportResult
            Containing flux, multiplication factor, and any
            information needed by reduced order solver and
            depletion manager. These will be declared through
            :meth:`setHooks`

        """

    @abstractmethod
    def bosUpdate(self, compositions, timestep, power) -> None:
        """Perform any updating with new compositions

        Assume that new materials have already been updated
        in memory

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
        """

    def bosSolve(self, newComps, timestep, power) -> TransportResult:
        """Solve BOS transport and return results

        Relies upon :meth:`bosUpdate`, :meth:`bosExecute`,
        :meth:`bosProcessResults` and :meth:`bosFinalize`

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
        self.bosUpdate(newComps, timestep, power)
        return super()._solve()

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
        difference = solver.needs.difference(self.features)
        if difference:
            raise IncompatibityError(
                "Cannot couple {hf} to {slv}.\n{slv} needs: {needs}\n"
                "{hf} has: {has}\nMissing: {diff}".format(
                    hf=self.__class__.__name__,
                    slv=solver.__class__.__name__,
                    needs=", ".join(sorted(solver.needs)),
                    has=", ".join(sorted(self.features)),
                    diff=", ".join(difference),
                )
            )


class ReducedOrderSolver(TransportSolver):
    """Reduced order solver"""

    @property
    @abstractmethod
    def needs(self) -> FeatureCollection:
        """Features needed by this reduced order solver"""

    @abstractmethod
    def substepUpdate(self, txResult, timestep, power) -> None:
        """Process information from latest transport result

        Parameters
        ----------
        txResult : hydep.TransportResult
            Result from latest high fidelity solution **or** reduced
            order solution. If ``timestep.substep == 0``, then ``txResult``
            will come from high fidelity solution
        timestep : hydep.internal.TimeStep
            Time step information
        power : float
            Current power [W]
        """

    def substepSolve(self, txResult, timestep, power) -> TransportResult:
        """Solve reduced order problem at a substep using previous results

        This method relies upon :meth:`substepUpdate`, :meth:`execute`,
        :meth:`processResults`, and :meth:`finalize`

        Parameters
        ----------
        txResult : hydep.TransportResult
            Result from previous solution. Will correspond to a high fidelity
            solution if ``timestep.substep == 0``.
        timestep : hydep.internal.TimeStep
            Current time step
        power : float
            Current power [W]

        Returns
        -------
        TransportResult
            At least flux, keff, and run time for this substep solution

        """
        self.substepUpdate(txResult, timestep, power)
        return super()._solve()

    def processBOS(self, txResult, timestep):
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

        """
