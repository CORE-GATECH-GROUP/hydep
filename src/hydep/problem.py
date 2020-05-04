"""
Primary class for handling geometry and material information
"""
import numbers
import logging
from collections.abc import Mapping
import copy
import pathlib
import os
import configparser
import typing
import tempfile

import numpy

from .constants import SECONDS_PER_DAY
from hydep.lib import HighFidelitySolver, ReducedOrderSolver, BaseStore
from hydep import Model, Manager, FailedSolverError
from .settings import Settings
from hydep.typed import TypedAttr
from hydep.internal import (
    TimeStep,
    compBundleFromMaterials,
    XsTimeMachine,
)

__logger__ = logging.getLogger("hydep")


class Problem(object):

    model = TypedAttr("model", Model)
    hf = TypedAttr("hf", HighFidelitySolver)
    rom = TypedAttr("rom", ReducedOrderSolver)
    dep = TypedAttr("dep", Manager)

    def __init__(self, model, hf, rom, dep, store=None):
        self._locked = False
        hf.checkCompatibility(dep)
        hf.checkCompatibility(rom)
        self.model = model
        self.hf = hf
        self.rom = rom
        self.dep = dep
        self.store = store
        self.settings = Settings()

    @property
    def store(self):
        return self._store

    @store.setter
    def store(self, value):
        if self._locked:
            raise AttributeError(f"{self} is locked and should not be modified")
        if value is None:
            self._store = None
            return
        if not isinstance(value, BaseStore):
            raise TypeError(f"Store should be subclass of {BaseStore}, not {value}")
        self._store = value

    def beforeMain(self):
        """Perform the necessary setup steps before the main sequence

        Not required to be called by the user, and may go private.
        Called during :meth:`solve`.

        Will create directories specified by
        :attr:`hydep.settings.Settings.basedir` and
        :attr:`hydep.settings.Settings.rundir` using
        :meth:`pathlib.Path.mkdir`, making parent directories as
        necessary. If ``rundir`` is ``None``, it will be assigned
        as the base directory.

        """
        __logger__.debug("Executing pre-solution routines")

        self.dep.beforeMain(self.model, self.settings)
        self.hf.beforeMain(self.model, self.dep, self.settings)
        self.rom.beforeMain(self.model, self.dep, self.settings)

        if self.store is None:
            from .hdfstore import HdfStore

            filename = self.settings.basedir / "hydep-results.h5"
            __logger__.debug(f"Storing result in {filename}")

            self.store = HdfStore(filename=filename)

        self.store.beforeMain(
            nhf=len(self.dep.timesteps) + 1,
            ntransport=sum(self.dep.substeps) + 1,
            ngroups=1,
            isotopes=tuple(self.dep.chain),
            burnableIndexes=[(m.id, m.name) for m in self.dep.burnable],
        )

    def solve(self, initialDays=0):
        """Launch the coupled sequence and hold your breath

        Parameters
        ----------
        initialDays : float, optional
            Non-negative number indicating the starting day. Defaults to zero.
            Useful for jumping into the middle of a schedule.

        """
        if not isinstance(initialDays, numbers.Real):
            raise TypeError(f"{type(initialDays)}")
        if initialDays < 0:
            raise ValueError(f"{initialDays}")

        self.hf.setHooks(self.dep.needs.union(self.rom.needs))

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
            self.rom.finalize(success)
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

        # TODO Some try / except around solutions? Or this whole method?
        # We want to make sure that all solvers are adequately warned about
        # failures
        __logger__.info(
            f"Executing {self.hf.__class__.__name__} step 0 "
            f"Time {timestep.currentTime / SECONDS_PER_DAY:.4E} [d]"
        )
        result = self.hf.bosSolve(compositions, timestep, self.dep.powers[0])
        __logger__.info(f"   k =  {result.keff[0]:.6f} +/- {result.keff[1]:.6E}")
        self.store.postTransport(timestep, result)
        if numpy.less(result.flux, 0).any():
            raise FailedSolverError(f"Negative fluxes obtained at {timestep}")
        self.rom.processBOS(result, timestep, self.dep.powers[0])

        xsmanager = XsTimeMachine(
            self.settings.fittingOrder,
            [startSeconds],
            [result.microXS],
            None if self.settings.unboundedFitting else self.settings.numFittingPoints,
        )

        fissionYields = result.fissionYields

        dtSeconds = self.dep.timesteps[0] / self.dep.substeps[0]
        result, compositions = self._marchSubstep(
            timestep, xsmanager, result, fissionYields, dtSeconds, compositions
        )

        # Go from second step to final step

        for coarseIndex, (coarseDT, power) in enumerate(
            zip(self.dep.timesteps[1:], self.dep.powers[1:]), start=1
        ):
            __logger__.info(
                f"Executing {self.hf.__class__.__name__} step {coarseIndex} "
                f"Time {timestep.currentTime / SECONDS_PER_DAY:.4E} [d]"
            )
            result = self.hf.bosSolve(compositions, timestep, power)
            __logger__.info(f"   k =  {result.keff[0]:.6f} +/- {result.keff[1]:.6E}")
            self.rom.processBOS(result, timestep, power)
            self.store.postTransport(timestep, result)

            xsmanager.append(timestep.currentTime, result.microXS)

            # Substeps
            # TODO Zip all three?
            dtSeconds = coarseDT / self.dep.substeps[coarseIndex]
            result, compositions = self._marchSubstep(
                timestep,
                xsmanager,
                result,
                result.fissionYields,
                dtSeconds,
                compositions,
            )

        # Final transport solution
        __logger__.info(
            f"Executing {self.hf.__class__.__name__} step {timestep.coarse} "
            f"Time {timestep.currentTime / SECONDS_PER_DAY:.4E} [d]"
        )
        result = self.hf.eolSolve(compositions, timestep, self.dep.powers[-1])
        __logger__.info(f"   k =  {result.keff[0]:.6f} +/- {result.keff[1]:.6E}")
        self.store.postTransport(timestep, result)
        if numpy.less(result.flux, 0).any():
            raise FailedSolverError(f"Negative fluxes obtained at {timestep}")

    def _marchSubstep(
        self, timestep, xsmachine, result, fissionYields, substepDT, compositions
    ):
        """March across the entire coarse step, using substeps if applicable

        ``timestep`` will be modified in place to mark the beginning
        of the next depletion interval. Results and compositions will
        be written to the current :attr:`store` when required.

        Parameters
        ----------
        timestep : hydep.internal.TimeStep
            Time step for the beginning of this coarse step
        xsmachine : hydep.internal.XsTimeMachine
            Micrscopic cross section and reaction rate manager
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

            rxnRates = xsmachine.getReactionRatesAt(timestep.currentTime, result.flux)
            compositions = self.dep.deplete(
                substepDT, compositions, rxnRates, fissionYields
            )

            timestep += substepDT
            self.store.writeCompositions(timestep, compositions)
            microXS = xsmachine.getMicroXsAt(timestep.currentTime)

            __logger__.info(
                f"Executing {self.rom.__class__.__name__} for substep {substepIndex}"
            )
            result = self.rom.substepSolve(timestep, compositions, microXS)
            if not numpy.isnan(result.keff).all():
                __logger__.info(
                    f"   k =  {result.keff[0]:.6f} +/- {result.keff[1]:.6E}"
                )
            self.store.postTransport(timestep, result)
            if numpy.less(result.flux, 0).any():
                raise FailedSolverError(f"Negative fluxes obtained at {timestep}")

        rxnRates = xsmachine.getReactionRatesAt(timestep.currentTime, result.flux)
        compositions = self.dep.deplete(
            substepDT, compositions, rxnRates, fissionYields
        )
        timestep.increment(substepDT, coarse=True)
        self.store.writeCompositions(timestep, compositions)

        return result, compositions

    def configure(
        self,
        options: typing.Union[
            str,
            pathlib.Path,
            typing.Mapping[str, typing.Any],
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
                    "archive on success": True,
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
            archive on success = 1

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
