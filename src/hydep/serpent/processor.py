"""
Class responsible for processing Serpent outputs
"""
import warnings

import serpentTools

from hydep.internal import TransportResult
from .fmtx import parseFmtx


__all__ = ["SerpentProcessor"]


class SerpentProcessor:
    """Class that pulls information from Serpent output files

    Requires the ``serpentTools`` package to interface with files
    https://github.com/CORE-GATECH-GROUP/serpent-tools

    Parameters
    ----------
    burnable : Optional[Tuple[str]]
        Names of universes with burnable materials. If not given,
        must be provided prior to processing files

    Attributes
    ----------
    burnable : Union[Tuple[str], None]
        Universes that contain burnable materials. Will attempt to
        pull data from these universes, including group fluxes,
        macroscopic cross sections, microscopic cross sections, and
        properties of the fission matrix. Must be set prior to
        processing files.
    options : dictionary
        Readable dictionary of settings to be applied to specific
        readers. The first level correspond to file types, e.g.
        ``resutls``, ``microxs``. The second level are key, value
        pairs of valid ``serpentTools`` settings to be set prior
        to file reading

    """

    options = {
        "results": {
            "serpentVersion": "2.1.31",
            "xs.variableGroups": ["xs", "diff-coeff"],
            "xs.variableExtras": [
                "GC_UNIVERSE_NAME",
                "INF_FLX",
                "INF_KINF",
                "MACRO_E",
                "MACRO_NG",
                "ABS_KEFF",
            ],
            "xs.getInfXS": True,
            "xs.getB1XS": False,
        },
        "microxs": {"microxs.getFlx": False},
    }

    def __init__(self, burnable=None):
        self.burnable = burnable

    @staticmethod
    def _warnOptions(settings, reason):
        pairs = ("{}: {}".format(k, v) for k, v in settings.items())
        msg = (
            "The following settings could not be used to control "
            "Serpent reading:\n{}\nReason: {}".format("\n  ".join(pairs), reason)
        )
        warnings.warn(msg, SyntaxWarning)

    def read(self, readable, filetype):
        """Read an output file and return the corresponding reader

        Wrapper around ``serpentTools.read`` with some extra sugar.
        First, and most plainly, if no options are pre-defined
        in :attr:`options` for a given file type, then nothing fancy
        is done.  The file is read and the resulting Reader is
        returned. Otherwise, the settings are passed to
        ``serpentTools`` and reverted after the file is read.

        Parameters
        ----------
        readable : str
            Name of the file to be read. There should be a
            corresponding ``serpentTools`` reader that can handle
            this file.
        filetype : str
            Type of the file, e.g. ``"results"``. While this is
            not necessary on the ``serpentTools`` side, it is used
            to pull pre-defined options to improve parsing. This
            is especially useful for the result file as only
            global multiplication factor and locally homogenized
            cross sections will be read.

        Returns
        -------
        Reader
            serpentTools reader that processed this file

        Warns
        -----
        SyntaxWarning
            Will be raised if setting any settings are unable to be
            passed into the settings manager. The file will be read
            regardless, but a warning will be raised

        """
        opts = self.options.get(filetype, {})
        if not opts:
            return serpentTools.read(readable, filetype)

        valuefails = {}
        unexpectedfails = {}

        with serpentTools.settings.rc as temp:
            for k, v in opts.items():
                try:
                    temp[k] = v
                except (KeyError, TypeError):
                    valuefails[k] = v
                except Exception:
                    unexpectedfails[k] = v
            serpentFile = serpentTools.read(readable, filetype)

        if valuefails:
            self._warnOptions(valuefails, "Bad settings and/or values")
        if unexpectedfails:
            self._warnOptions(
                unexpectedfails,
                "Unexpected failure with rc settings. File parsing still successful",
            )

        return serpentFile

    def processResult(self, resultfile, reqXS):
        """Scrape fluxes, multiplication factor, and xs

        Parameters
        ----------
        resultfile : str
            Serpent result file to be read
        reqXS : Iterable[str]
            Keys indicating what macroscopic cross sections should be
            pulled from each burnable universe. Fluxes will be taken
            automatically, so this should include other arguments
            like ``"abs"`` for absorption cross section, ``"fiss"``
            for fission, etc.

        Returns
        -------
        hydep.internal.TransportResult
            Bundling of multiplication factor and associated
            uncertainty, fluxes in burnable regions, and homogenized
            macroscopic cross sections in burnable regions.

        Raises
        ------
        AttributeError
            If :attr:`burnable` has not been set, indicating what
            universes to process, and in what order

        """
        if self.burnable is None:
            raise AttributeError(
                "No burnable universes present on {}".format(self.__class__.__name__)
            )

        results = self.read(resultfile, "results")

        keff = results.resdata["absKeff"]
        if len(keff.shape) == 2:
            keff = keff[0]
        keff[1] *= keff[0]

        xsLeader = "inf" if self.options["results"]["xs.getInfXS"] else "b1"

        allFluxes = []
        allXS = []

        # Take fluxes and cross sections from first step

        for univKey in self.burnable:
            xsdata = {}
            universe = results.getUniv(univKey, index=0)
            source = getattr(universe, xsLeader + "Exp")
            allFluxes.append(source[xsLeader + "Flx"])
            for reqKey in reqXS:
                xsdata[reqKey] = source[self._mapMacroXS(xsLeader, reqKey)]
            allXS.append(xsdata)

        return TransportResult(allFluxes, keff, macroXS=allXS)

    @staticmethod
    def _mapMacroXS(leader, name):
        return leader + name.capitalize()

    def processFmtx(self, fmtxfile):
        """Obtain the fission matrix from the outputs

        Parameters
        ----------
        fmtxfile : str
            Path to file containing fission matrix data.

        Returns
        -------
        scipy.sparse.csrmatrix
            Fission matrix ordered identically to :attr:`burnable`

        Raises
        ------
        AttributeError
            If :attr:`burnable` is not set
        ValueError
            If the ordering of the matrix is not consistent with
            :attr:`burnable`

        """
        if self.burnable is None:
            raise AttributeError(
                "No burnable universes present on {}".format(self.__class__.__name__)
            )

        with open(fmtxfile, "r") as stream:
            data = parseFmtx(stream)
        if data.universes != self.burnable:
            raise ValueError(
                "Universes are not ordered like burnable. May need to "
                "reshuffle - TBD")
        return data.matrix
