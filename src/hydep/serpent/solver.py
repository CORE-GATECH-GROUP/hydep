"""
The Serpent solver!
"""

import time
import tempfile
import shutil
import pathlib

import hydep
from hydep.internal import configmethod, TransportResult
import hydep.internal.features as hdfeat

from .writer import SerpentWriter
from .runner import SerpentRunner


class SerpentSolver(hydep.HighFidelitySolver):
    """Primary entry point for using Serpent as high fidelity solver

    Configuration should be done through the :meth:`configure` method.

    Attributes
    ----------
    features : hydep.internal.features.FeatureCollection
        Capabilities employed by this code that are relevant for
        this sequence. Can get basically all the macro XS possible,
        but also not sure how this part of the interface will
        go.
    """

    def __init__(self):
        self._curfile = None
        self._tmpdir = None
        self._writer = SerpentWriter()
        self._runner = SerpentRunner()

    @property
    def features(self):
        return hdfeat.FeatureCollection((
            hydep.features.FISSION_MATRIX,
            hydep.features.HOMOG_GLOBAL,
            hydep.features.HOMOG_LOCAL,
            hydep.features.MICRO_REACTION_XS,
        ), (True))

    @configmethod
    def configure(self, config):
        """Configure this interface

        Passes configuration data onto the :class:`SerpentWriter`.
        Settings are processed according to the following sections
        or keys:

            1. ``hydep`` - Global settings
            2. ``hydep.montecarlo`` - MC specific settings like particle
               statistics
            3. ``hydep.serpent`` - Serpent specific settings like
               executable path and cross section libraries.

        Settings found in later sections will overwrite those found in
        earlier sections.

        Parameters
        ----------
        config : Union[str, Mapping, pathlib.Path, configparser.ConfigParser]
            Configuration options to be processed. If ``str`` or
            ``pathlib.Path``, assume a file and read using
            :meth:`configparser.ConfigParser.read_file`. If a ``dict`` or
            other ``Mapping``, process with
            :meth:`configparser.ConfigParser.read_dict`. Otherwise load
            settings directly off the :class:`configparser.ConfigParser`
        """
        # TODO This could probably be polished up
        for level, path in enumerate(["hydep", "hydep.montecarlo", "hydep.serpent"]):
            if config.has_section(path):
                section = config[path]
                self._writer.configure(section, level)
                if level > 1:
                    self._runner.configure(section)

    def bosUpdate(self, _compositions, timestep):
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
       """
        self._curfile = self._writer.writeSteadyStateFile(
            "./serpent/s{}".format(timestep.coarse), timestep)

    def setHooks(self, needs):
        self._writer.hooks = needs

    def execute(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp = pathlib.Path(self._tmpdir.name) / self._curfile.name
        shutil.move(self._curfile, tmp)

        start = time.time()
        self._runner(tmp)

        return time.time() - start

    def processResults(self):
        # TODO THIS
        res = TransportResult(None, None)
        return res

    def finalize(self, _status):
        # TODO Zip on failure
        self._tmpdir.cleanup()

    def beforeMain(self, model, orderedBumat):
        self._writer.model = model
        self._writer.burnable = orderedBumat
        self._writer.writeBaseFile("./serpent/base.sss")

