"""
The Serpent solver!
"""

import hydep
import hydep.features
from hydep.internal import configmethod

from .writer import SerpentWriter


class SerpentSolver(hydep.HighFidelitySolver):
    """Primary entry point for using Serpent as high fidelity solver

    Configuration should be done through the :meth:`configure` method.

    Attributes
    ----------
    features : frozenset of hydep.features.Feature
        Capabilities employed by this code that are relevant for
        this sequence
    """

    def __init__(self):
        self._writer = SerpentWriter()

    @property
    def features(self):
        return frozenset((
            hydep.features.FISSION_MATRIX,
            hydep.features.HOMOG_GLOBAL,
            hydep.features.HOMOG_LOCAL,
            hydep.features.MICRO_REACTION_XS,
        ))

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
        for level, path in enumerate(["hydep", "hydep.montecarlo", "hydep.serpent"]):
            if config.has_section(path):
                self._writer.configure(config, path, level)

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
        self._writer.writeSteadyStateFile("./trial/s0", timestep)

    def setHooks(self, needs):
        self._writer.hooks.update(needs)

    def execute(self):
        pass

    def processResults(self):
        pass

    def finalize(self):
        pass

    def beforeMain(self, model, orderedBumat):
        self._writer.model = model
        self._writer.burnable = orderedBumat
        self._writer.writeBaseFile("./trial/serpent.sss")

