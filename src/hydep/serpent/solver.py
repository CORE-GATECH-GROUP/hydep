"""
The Serpent solver!
"""

from abc import abstractmethod
import time
import tempfile
import shutil
import pathlib
import zipfile
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

    """

    def __init__(self, writer, runner, processor=None):
        self._writer = writer
        self._processor = processor or SerpentProcessor()
        self._runner = runner
        self._hooks = None
        self._volumes = None

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
    def writer(self) -> BaseWriter:
        return self._writer

    @property
    def runner(self) -> BaseRunner:
        return self._runner

    @property
    def processor(self) -> SerpentProcessor:
        return self._processor

    def _process(self, basefile, index=0):
        if self.hooks is not None and self.hooks.macroXS:
            res = self.processor.processResult(
                basefile + "_res.m",
                self.hooks.macroXS,
                index=index,
            )
        else:
            keff = self.processor.getKeff(basefile + "_res.m", index=index)
            fluxes = self.processor.processDetectorFluxes(
                basefile + f"_det{index}.m",
                "flux",
            )
            res = TransportResult(fluxes, keff)

        res.flux = res.flux / self._volumes

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
        settings : hydep.settings.HydepSettings
            Shared settings

        """
        self.runner.configure(settings.serpent)

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

        __logger__.debug("Writing base Serpent input file")
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
        self._archiveOnSuccess = False

    def bosUpdate(self, compositions, timestep, power):
        """Create a new input file with updated compositions

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
       """
        self._curfile = self.writer.writeSteadyStateFile(
            f"./serpent/s{timestep.coarse}", compositions, timestep, power
        )

    def eolUpdate(self, compositions, timestep, power):
        """Write the final input file

        Nearly identical to the file generated by :meth:`bosUpdate`, except
        no depletion will be provided.

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
        self._curfile = self.writer.writeSteadyStateFile(
            f"./serpent/s{timestep.coarse}", compositions, timestep, power, final=True
        )

    def execute(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmpFile = pathlib.Path(self._tmpdir.name) / self._curfile.name
        shutil.move(self._curfile, self._tmpFile)

        start = time.time()
        self.runner(self._tmpFile)

        return time.time() - start

    def processResults(self):
        """Pull necessary information from Serpent outputs

        Returns
        -------
        TransportResult
            At the very least a result containing the flux in each burnable
            region and multiplication factor. Other data will be attached
            depending on the :attr:`hooks`, including fission matrix
            and macrosopic cross sections

        Raises
        ------
        AttributeError
            If the ordering of burnable materials is not known. This
            should be set prior to :meth:`execute`, typically in
            :meth:`bosUpdate` or :meth:`beforeMain`

        """
        return self._process(str(self._tmpFile), index=0)

    def finalize(self, status):
        if self._curfile is not None and (self._archiveOnSuccess or not status):
            self._archive()
        self._tmpdir.cleanup()

    def _archive(self):
        skipExts = {".seed", ".out", ".dep"}
        zipf = self._curfile.with_suffix(".zip")

        __logger__.debug(f"Archiving Serpent results to {zipf.resolve()}")

        with zipfile.ZipFile(zipf, "w") as myzip:
            for ff in self._tmpFile.parent.glob("*"):
                for ext in skipExts:
                    if ff.name.endswith(ext):
                        break
                else:
                    myzip.write(ff, ff.name)

    def _writeMainFile(self, model, manager, settings):
        basefile = settings.rundir / "serpent" / "base.sss"
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
        self._fp = basefile = settings.rundir / "serpent-extdep" / "external"
        self.writer.writeCouplingFile(
            basefile,
            manager.timesteps,
            manager.powers,
            settings,
        )
        return self._fp

    def bosUpdate(self, compositions, timestep, _power):
        # Skip updating for step 0 as BOL compositions aready written
        if timestep.coarse == 0:
            return
        self._cstep = timestep.coarse
        self.writer.updateComps(compositions, timestep, threshold=0)

    def eolUpdate(self, compositions, timestep, _power):
        self._cstep = -timestep.coarse
        self.writer.updateComps(compositions, timestep, threshold=0)

    def execute(self):
        if self._cstep > 0:
            self.runner.solveNext()
        elif self._cstep == 0:
            self.runner.start(self._fp, self._fp.with_suffix(".log"))
            self.writer.updateFromRestart()
        else:
            self.runner.solveEOL()

    def processResults(self):
        """Pull necessary information from Serpent outputs

        Returns
        -------
        TransportResult
            At the very least a result containing the flux in each burnable
            region and multiplication factor. Other data will be attached
            depending on the :attr:`hooks`, including fission matrix
            and macrosopic cross sections

        Raises
        ------
        AttributeError
            If the ordering of burnable materials is not known.

        """
        step = self._cstep if self._cstep >= 0 else -self._cstep
        base = str(self._fp)
        return self._process(base, step)
