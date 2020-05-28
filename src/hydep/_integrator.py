from abc import ABC, abstractmethod
import logging
import numbers
from collections.abc import Mapping
import copy
import pathlib
import os
import configparser
import typing
import tempfile

import numpy

from .exceptions import FailedSolverError
from .model import Model
from .manager import Manager
from .settings import Settings
from .lib import HighFidelitySolver, ReducedOrderSolver, BaseStore
from .typed import TypedAttr
from .constants import SECONDS_PER_DAY
from .internal import DataBank, compBundleFromMaterials, TimeStep

__logger__ = logging.getLogger("hydep")


class Integrator(ABC):
    """Base class for time integration

    Inspired by `OpenMC integrators
    <https://docs.openmc.org/en/latest/pythonapi/generated/openmc.deplete.
    abc.Integrator.html>`_

    Responsible for marching forward in time, but using reduced order
    solutions for the intermediate, non-BOS solutions. Here, BOS can
    refer to the beginning of a coarse step, a high-fidelity transport
    solution, or substep following a reduced order solution.

    The BOS solution will be performed outside this class, and
    the reaction rates will be provided at the beginning of the solution.

    Parameters
    ----------
    model : hydep.Model
        Representation of the problem geometry
    hf : hydep.lib.HighFidelitySolver
        High fidelity solver to be executed at the beginning
        of each coarse step, and the final EOL point
    ro : hydep.lib.ReducedOrderSolver
        Reduced order solver to be executed at the substeps,
        and potentially any intermediate time points
    dep : hydep.Manager
        Depletion manager, including access to depletion chain
        and depletion solver
    store : hydep.lib.BaseStore, optional
        Instance responsible for writing transport and depletion
        result data. If not provided, will be set to
        :class:`hydep.hdfstore.HdfStore`

    Attributes
    ----------
    model : hydep.Model
        Representation of the problem geometry
    hf : hydep.lib.HighFidelitySolver
        High fidelity solver to be executed at the beginning
        of each coarse step, and the final EOL point
    ro : hydep.lib.ReducedOrderSolver
        Reduced order solver to be executed at the substeps,
        and potentially any intermediate time points
    dep : hydep.Manager
        Depletion manager, including access to depletion chain
        and depletion solver
    store : hydep.lib.BaseStore or None
        Instance responsible for writing transport and depletion
        result data. If not provided, will be set to
        :class:`hydep.hdfstore.HdfStore`
    settings : hydep.Settings
        Simulation settings. Can be updated directly, or
        through :meth:`configure`

    """

    model = TypedAttr("model", Model)
    hf = TypedAttr("hf", HighFidelitySolver)
    ro = TypedAttr("ro", ReducedOrderSolver)
    dep = TypedAttr("dep", Manager)

    def __init__(self, model, hf, ro, dep, store=None):
        self._locked = False
        hf.checkCompatibility(dep)
        hf.checkCompatibility(ro)
        self.model = model
        self.hf = hf
        self.ro = ro
        self.dep = dep
        self.store = store
        self.settings = Settings()
        self._xs = None

    @abstractmethod
    def __call__(
        self,
        tstart: "hydep.internal.TimeStep",
        dt: float,
        compositions: "hydep.internal.CompBundle",
        flux: numpy.ndarray,
        fissionYields: typing.List[
            typing.Dict[int, "hydep.internal.FissionYield"]
        ],
    ) -> "hydep.internal.CompBundle":
        """Progress across a single step given BOS data

        Parameters
        ----------
        tstart : float
            Current point in calendar time [s]
        dt : float
            Length of depletion interval [s]
        compositions : hydep.internal.CompBundle
            Beginning-of-step compositions for all burnable materials
        flux : numpy.ndarray
            Beginning-of-step flux in all burnable materials
        fissionYields : list of mapping {int: fission yield}
            Fission yields in all burnable materials, ordered consistently
            with with material ordering of ``flux`` and ``compositions``.
            Each entry is a dictionary mapping parent isotope ZAI
            to :class:`hydep.internal.FissionYield` for that isotope.
            Suitable to be passed directly off to
            :meth:`hydep.Manager.deplete`

        Returns
        -------
        hydep.internal.CompBundle
            End-of-step compositions in all burnable materials. Intermediate
            points should not be returned

        """

    @property
    def store(self):
        return self._store

    @store.setter
    def store(self, value):
        if self._locked:
            raise AttributeError(
                f"{self} is locked and should not be modified"
            )
        if value is None:
            self._store = None
            return
        if not isinstance(value, BaseStore):
            raise TypeError(
                f"Store should be subclass of {BaseStore}, not {value}"
            )
        self._store = value

    def beforeMain(self):
        """Perform the necessary setup steps before the main sequence

        Not required to be called by the user, and may go private.
        Called during :meth:`solve`.

        Will create directories specified by
        :attr:`hydep.Settings.basedir` and
        :attr:`hydep.Settings.rundir` using
        :meth:`pathlib.Path.mkdir`, making parent directories as
        necessary. If ``rundir`` is ``None``, it will be assigned
        as the base directory.

        """
        __logger__.debug("Executing pre-solution routines")

        self.dep.beforeMain(self.model, self.settings)
        self.hf.beforeMain(self.model, self.dep, self.settings)
        self.ro.beforeMain(self.model, self.dep, self.settings)

        if self.store is None:
            from .hdfstore import HdfStore

            filename = self.settings.basedir / "hydep-results.h5"
            __logger__.debug("Storing result in %s", filename)

            self.store = HdfStore(filename=filename)

        self.store.beforeMain(
            nhf=len(self.dep.timesteps) + 1,
            ntransport=sum(self.dep.substeps) + 1,
            ngroups=1,
            isotopes=tuple(self.dep.chain),
            burnableIndexes=[
                (m.id, m.name, m.volume) for m in self.dep.burnable
            ],
        )

        self._xs = DataBank(
            self.settings.numFittingPoints,
            len(self.dep.burnable),
            self.dep.chain.reactionIndex,
            self.settings.fittingOrder,
        )

    def solve(self, initialDays=0):
        """Alias for :meth:`integrate` to work with old scripts"""
        import warnings
        warnings.warn("Use integrate instead of solve")
        return self.integrate(initialDays=initialDays)

    def integrate(self, initialDays=0):
        """Launch the coupled sequence and hold your breath

        Parameters
        ----------
        initialDays : float, optional
            Non-negative number indicating the starting day. Defaults
            to zero.  Useful for jumping into the middle of a schedule.
            Primarily for cosmetic changes (e.g. logging, storing in
            :attr:`store`)

        Raises
        ------
        hydep.FailedSolverError
            If one of the transport solutions obtains a negative flux.
            Flux will be stored prior to failure
        hydep.GeometryError
            If the geometry is not well configured, e.g. unbounded in
            more than one dimension
        hydep.NegativeDensityError
            If substantially negative densities are obtained from
            :attr:`hydep.Manager.deplete`
        hydep.IncompatibilityError
            If :attr:`hf` is not compatible with :attr:`dep` or
            :attr:`ro`

        """
        if not isinstance(initialDays, numbers.Real):
            raise TypeError(f"{type(initialDays)}")
        if initialDays < 0:
            raise ValueError(f"{initialDays}")

        self.hf.setHooks(self.dep.needs.union(self.ro.needs))

        self.settings.validate()

        # Check directories, making them as necessary
        basedir = self.settings.basedir
        if not basedir.is_dir():
            basedir.mkdir(parents=True)

        rundir = self.settings.rundir
        tempdir = None
        if rundir is None:
            if self.settings.useTempDir:
                tempdir = tempfile.TemporaryDirectory()
                self.settings.rundir = pathlib.Path(tempdir.name)
            else:
                self.settings.rundir = basedir
        elif not rundir.is_dir():
            rundir.mkdir(parents=True)

        previousDir = pathlib.Path.cwd()

        success = False

        try:
            os.chdir(self.settings.rundir)
            self.beforeMain()

            # Context manager?
            self._locked = True
            self._mainsequence(initialDays * SECONDS_PER_DAY)
            success = True
        finally:
            self.hf.finalize(success)
            self.ro.finalize(success)
            self._locked = False
            os.chdir(previousDir)
            if tempdir is not None:
                tempdir.cleanup()
                self.settings.rundir = None

    def _mainsequence(self, startSeconds):
        compositions = compBundleFromMaterials(
            self.dep.burnable, tuple(self.dep.chain)
        )
        timestep = TimeStep(currentTime=startSeconds)
        self.store.writeCompositions(timestep, compositions)

        # Run first solution to get information on micro xs

        __logger__.info(
            "Executing %s step 0 Time %.4E [d]",
            type(self.hf).__name__, startSeconds / SECONDS_PER_DAY,
        )
        result = self.hf.bosSolve(compositions, timestep, self.dep.powers[0])
        __logger__.info("   k =  %.6f +/- %.6E", result.keff[0], result.keff[1])
        self.store.postTransport(timestep, result)
        if numpy.less(result.flux, 0).any():
            raise FailedSolverError(f"Negative fluxes obtained at {timestep}")
        self.ro.processBOS(result, timestep, self.dep.powers[0])

        self._xs.push(startSeconds, result.microXS)

        fissionYields = result.fissionYields

        dtSeconds = self.dep.timesteps[0] / self.dep.substeps[0]
        result, compositions = self._marchSubstep(
            timestep, self._xs, result, fissionYields, dtSeconds, compositions
        )

        # Go from second step to final step

        for coarseIndex, (coarseDT, power) in enumerate(
            zip(self.dep.timesteps[1:], self.dep.powers[1:]), start=1
        ):
            __logger__.info(
                "Executing %s step %d Time %.4E [d]",
                type(self.hf).__name__, coarseIndex, timestep.currentTime / SECONDS_PER_DAY,
            )
            result = self.hf.bosSolve(compositions, timestep, power)
            __logger__.info("   k =  %.6f +/- %.6E", result.keff[0], result.keff[1])
            self.ro.processBOS(result, timestep, power)
            self.store.postTransport(timestep, result)

            self._xs.push(timestep.currentTime, result.microXS)

            # Substeps
            # TODO Zip all three?
            dtSeconds = coarseDT / self.dep.substeps[coarseIndex]
            result, compositions = self._marchSubstep(
                timestep,
                self._xs,
                result,
                result.fissionYields,
                dtSeconds,
                compositions,
            )

        # Final transport solution
        __logger__.info(
            "Executing %s step %d Time %.4E [d]",
            type(self.hf).__name__, timestep.coarse, timestep.currentTime / SECONDS_PER_DAY,
        )
        result = self.hf.eolSolve(compositions, timestep, self.dep.powers[-1])
        __logger__.info("   k =  %.6f +/- %.6E", result.keff[0], result.keff[1])
        self.store.postTransport(timestep, result)
        if numpy.less(result.flux, 0).any():
            raise FailedSolverError(f"Negative fluxes obtained at {timestep}")

    def _marchSubstep(
        self,
        timestep,
        xsmachine,
        result,
        fissionYields,
        substepDT,
        compositions,
    ):
        """March across the entire coarse step, using substeps if applicable

        ``timestep`` will be modified in place to mark the beginning
        of the next depletion interval. Results and compositions will
        be written to the current :attr:`store` when required.

        Parameters
        ----------
        timestep : hydep.internal.TimeStep
            Time step for the beginning of this coarse step
        xsmachine : hydep.internal.DataBank
            Microscopic cross section and reaction rate manager
        result : hydep.internal.TransportResult
            Most recent transport result
        fissionYields : sequence of mapping int -> FissionYield
            Beginning of step fission yields. Will be updated at
            each substep if the reduced order solver updates the
            fission yields
        substepDT : float
            Length of time [s] for each substep
        compositions : hydep.internal.CompBundle
            Compositions from the beginning of the time step

        Returns
        -------
        hydep.internal.TransportResult
            Result from the last reduced order simulation, if one was
            performed. Otherwise will be identical to the input
        hydep.internal.CompBundle
            Compositions for the beginning of the next step

        """
        for substepIndex in range(self.dep.substeps[timestep.coarse] - 1):
            if substepIndex and result.fissionYields is not None:
                fissionYields = result.fissionYields

            compositions = self(
                timestep, substepDT, compositions, result.flux, fissionYields,
            )

            timestep += substepDT
            self.store.writeCompositions(timestep, compositions)
            microXS = xsmachine.at(timestep.currentTime)

            __logger__.info(
                "Executing %s for substep %d",
                type(self.ro).__name__, substepIndex,
            )
            result = self.ro.substepSolve(timestep, compositions, microXS)
            if not numpy.isnan(result.keff).all():
                __logger__.info("   k =  %.6f +/- %.6E", result.keff[0], result.keff[1])
            self.store.postTransport(timestep, result)
            if numpy.less(result.flux, 0).any():
                raise FailedSolverError(
                    f"Negative fluxes obtained at {timestep}"
                )

        compositions = self(
            timestep, substepDT, compositions, result.flux, fissionYields,
        )
        timestep.increment(substepDT, coarse=True)
        self.store.writeCompositions(timestep, compositions)

        return result, compositions

    def configure(
        self,
        options: typing.Union[
            str, pathlib.Path, typing.Mapping[str, typing.Any],
        ],
    ):
        """Configure the associated solvers

        This action can be performed multiple times, but the most
        recently updated values will be used in the simulation. Must
        have at least a key or section ``hydep`` in the first level.
        Subsequent solvers, e.g. Serpent, can be configured using
        a sub-level ``"hydep.<solver>"``.

        The following would be appropriate to configure an associated
        Serpent solver::

            {
                "hydep": {
                    "fitting order": 1,
                    ...,
                },
                "hydep.serpent": {
                    "executable": "sss2",
                    ...
                },
            }

        A corresponding configuration file would be

        .. code:: INI

            [hydep]
            fitting order = 1

            [hydep.serpent]
            executable  = sss2

        Parameters
        ----------
        config : Union[str, Path, dict]
            If a string or :class:`pathlib.Path`, assume a INI / CFG
            file. If a dictionary or
            :class:`~collections.abc.Mapping`-like, assume the data
            contain the necessary structure and config options and
            read those in

        Raises
        ------
        TypeError
            If ``config`` is not understood

        """
        if isinstance(options, (str, pathlib.Path)):
            __logger__.debug(f"Reading config options from {options}")
            cfg = configparser.ConfigParser()
            with open(options, "r") as stream:
                cfg.read_file(stream, options)
            options = dict(cfg)
        elif not isinstance(options, Mapping):
            raise TypeError(
                f"Options must be a mapping-type or file, not {type(options)}"
            )
        else:
            options = copy.deepcopy(options)

        self.settings.updateAll(options)
