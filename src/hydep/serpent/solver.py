"""
The Serpent solver!
"""

from abc import abstractmethod
import time
import pathlib
import logging

import numpy
from hydep.lib import HighFidelitySolver
from hydep.internal import TransportResult
import hydep.internal.features as hdfeat

from .writer import BaseWriter, SerpentWriter, ExtDepWriter
from .runner import BaseRunner, SerpentRunner, ExtDepRunner
from .processor import SerpentProcessor, FissionYieldFetcher
from .xsavail import XS_2_1_30


__logger__ = logging.getLogger("hydep.serpent")

_FEATURES_ATLEAST_2_1_30 = hdfeat.FeatureCollection(
    (
        hdfeat.FISSION_MATRIX,
        hdfeat.FISSION_YIELDS,
        hdfeat.HOMOG_GLOBAL,
        hdfeat.HOMOG_LOCAL,
        hdfeat.MICRO_REACTION_XS,
    ),
    XS_2_1_30,
)


class BaseSolver(HighFidelitySolver):
    """Base solver for interfacing with Serpent >= 2.1.30

    Does not provide all methods needed by the
    :class:`hydep.lib.HighFidelitySolver` other than
    :attr:`features`, :meth:`setHooks`. :meth:`beforeMain`
    is partially implemented, but requires a helper method for
    writing solver-specific input files. Other methods for
    directly interacting with Serpent are left to concrete
    classes like :class:`SerpentSolver` and
    :class:`CoupledSerpentSolver`.

    Parameters
    ----------
    writer : hydep.serpent.BaseWriter
        Passed to :attr:`writer`
    runner : hydep.serpent.BaseRunner
        Passed to :attr:`runner`
    processor : hydep.serpent.SerpentProcessor, optional
        If not provided, use a :class:`hydep.serpent.SerpentProcessor`.
        Passed to :attr:`processor`

    Parameters
    ----------
    writer : hydep.serpent.BaseWriter
        Passed to :attr:`writer`
    runner : hydep.serpent.BaseRunner
        Passed to :attr:`runner`
    processor : hydep.serpent.SerpentProcessor, optional
        If not provided, use a :class:`hydep.serpent.SerpentProcessor`.
        Passed to :attr:`processor`

    Attributes
    ----------
    writer : hydep.serpent.BaseWriter
        Instance responsible for writing input files
    runner : hydep.serpent.BaseRunner
        Instance reponsible for controlling Serpent execution
    processor : hydep.serpent.SerpentProcessor
        Instance responsible for pulling data from output files
    features : hydep.internal.features.FeatureCollection
        A non-exhaustive list of features contained in Serpent >= 2.1.30
        that are useful / necessary for this framework
    hooks : hydep.internal.features.FeatureCollection
        Collection of physics and cross sections needed by other
        aspects of the framework
    basedir : pathlib.Path or None
        Path where results will be saved and auxillary files may be saved.
        Configured in :meth:`self.beforeMain`

    """

    def __init__(self, writer, runner, processor=None):
        self._writer = writer
        self._processor = processor or SerpentProcessor()
        self._runner = runner
        self._hooks = None
        self._volumes = None
        self._basedir = None

    @property
    def features(self):
        return _FEATURES_ATLEAST_2_1_30

    @property
    def hooks(self):
        return self._hooks

    def setHooks(self, needs):
        """Instruct the solver and helpers what physics are needed

        Parameters
        ----------
        needs : hydep.internal.features.FeatureCollection
            The needs of other solvers, e.g.
            :class:`hydep.ReducedOrderSolver`

        """
        # TODO Guard against hooks that aren't supported
        if not isinstance(needs, hdfeat.FeatureCollection):
            raise TypeError(
                "Hooks must be {}, not {}".format(
                    hdfeat.FeatureCollection.__name__, type(needs)
                )
            )

        self._hooks = needs
        self._writer.hooks = needs

    @property
    def basedir(self):
        return self._basedir

    @property
    def writer(self) -> BaseWriter:
        return self._writer

    @property
    def runner(self) -> BaseRunner:
        return self._runner

    @property
    def processor(self) -> SerpentProcessor:
        return self._processor

    def _process(self, basefile, index=0):
        fluxes = self.processor.processDetectorFluxes(
            basefile + f"_det{index}.m",
            "flux",
        ) / self._volumes

        if self.hooks is not None and self.hooks.macroXS:
            resbundle = self.processor.processResult(
                basefile + "_res.m",
                self.hooks.macroXS,
                index=index,
            )
            res = TransportResult(fluxes, resbundle.keff, macroXS=resbundle.macroXS)
        else:
            keff = self.processor.getKeff(basefile + "_res.m", index=index)
            res = TransportResult(fluxes, keff)

        if not self.hooks:
            return res

        for feature in self.hooks.features:
            if feature is hdfeat.FISSION_MATRIX:
                res.fmtx = self.processor.processFmtx(basefile + f"_fmtx{index}.m")
            elif feature is hdfeat.MICRO_REACTION_XS:
                res.microXS = self.processor.processMicroXS(basefile + f"_mdx{index}.m")
            elif feature is hdfeat.FISSION_YIELDS:
                res.fissionYields = self.processor.processFissionYields(
                    basefile + f"_det{index}.m"
                )

        return res

    def beforeMain(self, model, manager, settings):
        """Prepare the base input file

        Parameters
        ----------
        model : hydep.Model
            Geometry information to be written once
        manager : hydep.Manager
            Depletion information
        settings : hydep.Settings
            Shared settings

        """
        self.runner.configure(settings.serpent)
        self._basedir = settings.basedir

        assert manager.burnable is not None
        orderedBumat = manager.burnable

        matids = []
        self._volumes = numpy.empty((len(orderedBumat), 1))
        for ix, m in enumerate(orderedBumat):
            matids.append(str(m.id))
            self._volumes[ix] = m.volume

        self.writer.model = model
        self.writer.burnable = orderedBumat

        acelib = settings.serpent.acelib
        if acelib is None:
            raise AttributeError(
                f"Serpent acelib setting not configured on {settings}"
            )
        self.writer.updateProblemIsotopes((iso.triplet for iso in manager.chain), acelib)

        __logger__.info("Writing base Serpent input file")
        mainfile = self._writeMainFile(model, manager, settings)

        self.processor.burnable = matids

        # Not super pretty, as this interacts both with the writer's roles
        # and the processors roles
        if hdfeat.FISSION_YIELDS in self.hooks.features:
            fyproc = FissionYieldFetcher(matids, manager.chain)
            detlines = fyproc.makeDetectors(upperEnergy=20)
            if detlines:
                with mainfile.open("a") as s:
                    s.write("\n".join(detlines))
            self.processor.fyHelper = fyproc

        __logger__.info("Done.")

    @abstractmethod
    def _writeMainFile(self, model, manager) -> pathlib.Path:
        """Write the primary input file before transport solutions"""
        pass


class SerpentSolver(BaseSolver):
    """Primary entry point for using Serpent as high fidelity solver

    Attributes
    ----------
    writer : SerpentWriter
        Responsible for writing Serpent inputs
    runner : SerpentRunner
        Responsible for runing Serpent
    processor : SerpentProcessor
        Responsible for processing outputs
    features : hydep.internal.features.FeatureCollection
        A non-exhaustive list of features contained in Serpent >= 2.1.30
        that are useful / necessary for this framework
    hooks : hydep.internal.features.FeatureCollection
        Collection of physics and cross sections needed by other
        aspects of the framework

    """

    def __init__(self):
        super().__init__(
            writer=SerpentWriter(),
            runner=SerpentRunner(),
        )
        self._curfile = None
        self._tmpdir = None
        self._tmpFile = None

    def bosSolve(self, compositions, timestep, power):
        """Create and solve the BOS problem with updated compositions

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
            Current reactor power [W]

        Returns
        -------
        hydep.internal.TransportResult
            Transport result with fluxes, multiplication factor,
            run time, and other data needed in the framework

        """
        return self._solve(compositions, timestep, power, final=False)

    def eolSolve(self, compositions, timestep, power):
        """Create and solve the EOL problem with updated compositions

        The only difference between this and :meth:`bosSolve` is that
        burnable materials are not marked with ``burn 1``.

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
            Current reactor power [W]

        Returns
        -------
        hydep.internal.TransportResult
            Transport result with fluxes, multiplication factor,
            run time, and other data needed in the framework

        """
        return self._solve(compositions, timestep, power, final=True)

    def _solve(self, compositions, timestep, power, final=False):
        curfile = self.writer.writeSteadyStateFile(
            f"./serpent-s{timestep.coarse}", compositions, timestep, power, final=final)

        start = time.time()
        self.runner(curfile)
        end = time.time()

        res = self._process(str(curfile), index=0)
        res.runTime = end - start
        return res

    def _writeMainFile(self, model, manager, settings):
        basefile = pathlib.Path.cwd() / "serpent-base.sss"
        self.writer.writeBaseFile(basefile, settings)
        return basefile


class CoupledSerpentSolver(BaseSolver):
    """Utilize the external depletion interface

    Attributes
    ----------
    writer : hydep.serpent.ExtDepWriter
        Instance that writes the main input file and new compositions
    runner : hydep.serpent.ExtDepRunner
        Instance that communicates with a Serpent process. Controls how
        Serpent walks through time
    processor : hydep.serpent.SerpentProcessor
        Responsible for pulling information from output files
    features : hydep.internal.features.FeatureCollection
        A non-exhaustive list of features contained in Serpent >= 2.1.30
        that are useful / necessary for this framework
    hooks : hydep.internal.features.FeatureCollection
        Collection of physics and cross sections needed by other
        aspects of the framework

    """

    def __init__(self):
        super().__init__(
            writer=ExtDepWriter(),
            runner=ExtDepRunner(),
        )
        self._cstep = 0
        self._fp = None

    def _writeMainFile(self, model, manager, settings):
        self._fp = basefile = pathlib.Path.cwd() / "serpent-extdep"
        self.writer.writeCouplingFile(
            basefile,
            manager.timesteps,
            manager.powers,
            settings,
        )
        return self._fp

    def bosSolve(self, compositions, timestep, _power):
        """Solve the BOS problem

        For the first timestep, where ``timestep.coarse == 0``,
        the :attr:`runner` will initiate the Serpent process
        and solve the first point in time. Otherwise, the compositions
        are written to the restart file, which are read into Serpent
        using the external depletion interface.

        Parameters
        ----------
        compositions : hydep.internal.CompBundle
            Material compositions for this point in time.
        timestep : hydep.internal.TimeStep
            Current time step
        _power : float
            Current reactor power [W]. Not needed since the
            powers are already written to the full input file

        Returns
        -------
        hydep.internal.TransportResult
            Transport result with fluxes, multiplication factor,
            run time, and other information needed by the framework.

        """
        if timestep.coarse != 0:
            self.writer.updateComps(compositions, timestep, threshold=0)
            start = time.time()
            self.runner.solveNext()
            end = time.time()
        else:
            start = time.time()
            self.runner.start(self._fp, self._fp.with_suffix(".log"))
            end = time.time()
            self.writer.updateFromRestart()

        res = self._process(str(self._fp), timestep.coarse)
        res.runTime = end - start

        return res

    def eolSolve(self, compositions, timestep, _power):
        """Solve the EOL problem

        The only difference between this and :meth:`bosSolve`
        is that that :meth:`ExtDepRunner.solveEOL` is used to
        solve and then terminate the Serpent process.

        Parameters
        ----------
        compositions : hydep.internal.CompBundle
            Material compositions for this point in time.
        timestep : hydep.internal.TimeStep
            Current time step
        _power : float
            Current reactor power [W]. Not needed since the
            powers are already written to the full input file

        Returns
        -------
        hydep.internal.TransportResult
            Transport result with fluxes, multiplication factor,
            run time, and other information needed by the framework.

        """

        self.cstep = timestep.coarse
        self.writer.updateComps(compositions, timestep, threshold=0)
        start = time.time()
        self.runner.solveEOL()
        end = time.time()

        res = self._process(str(self._fp), timestep.coarse)
        res.runTime = end - start
        return res

    def finalize(self, _success):
        """Close the connection to the Serpent solver

        Parameters
        ----------
        _success : bool
            Success flag. Not used

        """
        self._runner.terminate()
