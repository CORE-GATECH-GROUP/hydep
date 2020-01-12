"""
Class responsible for processing Serpent outputs
"""
import warnings
import copy
from functools import wraps

import serpentTools

from hydep.internal import TransportResult, MicroXsVector, FakeSequence
from .fmtx import parseFmtx
from hydep.internal import allIsotopes


__all__ = ["SerpentProcessor"]


def requireBurnable(m):
    """Decorator that requires the processor's burnable to be set"""

    @wraps(m)
    def checkburnable(s, *args, **kwargs):
        if s.burnable is None:
            raise AttributeError(
                "{} requires burnable universes to be "
                "set on {}".format(m.__name__, s)
            )
        return m(s, *args, **kwargs)

    return checkburnable


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
        "fission yield energy": 0.0253,
    }
    fissionYieldEnergies = {
        "thermal": 0.0253,
        "epithermal": 5.0e5,
        "fast": 14e6,
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

    def getKeff(self, resultfile):
        """Pull just the multiplication factor from a result file

        Parameters
        ----------
        resultfile : str
            Path to the result file to be read

        Returns
        -------
        Iterable[float, float]
            Multiplication factor and absolute uncertainty

        """
        results = self.read(resultfile, "results")

        keff = results.resdata["absKeff"]
        if len(keff.shape) == 2:
            keff = keff[0]
        keff[1] *= keff[0]
        return keff

    @requireBurnable
    def processResult(self, resultfile, reqXS):
        """Scrape fluxes, multiplication factor, and xs

        Expects all the universes to have been selected for
        homogenization with the ``set gcu`` card. Will pull both
        fluxes and macroscopic cross sections from the file.
        If this is not the case, there will be errors. Examine
        some of the smaller methods that are used here to pull
        a single quantity

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

        See Also
        --------
        * :meth:`getKeff` - Just pull the multiplication factor and
           uncertainty
        * :meth:`processDetectorFluxes` - Pull fluxes in each burnable
          region from a detector file, rather than relying on
          homogenization

        """

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

    @requireBurnable
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

        with open(fmtxfile, "r") as stream:
            data = parseFmtx(stream)
        if data.universes != self.burnable:
            raise ValueError(
                "Universes are not ordered like burnable. May need to "
                "reshuffle - TBD"
            )
        return data.matrix

    def configure(self, section):
        """Configure the processor

        Looks for the following options:

        * ``"version"`` or ``"serpentVersion"`` [string] - set the
           version of Serpent used to generate results
        * ``useB1XS`` [boolean] - Pull cross sections from the
          B1 / critical leakage cross sections, e.g. ``"B1_ABS"``.
          Otherwise use infinite medium cross sections
        * ``fission yield energy`` {thermal, epithermal, fast} - Which energy
           group to use when pulling fission yields. Default is thermal

        Parameters
        ----------
        section : configparser.SectionProxy
            Sections of configuration that directly apply to Serpent,
            e.g. ``config["hydep.serpent"]``

        """

        if self.options is self.__class__.options:
            self.options = copy.deepcopy(self.__class__.options)

        version = section.get("version")

        if version is None:
            version = section.get("serpentVersion")

        if version is not None:
            if version not in {"2.1.31", "2.1.30", "2.1.29"}:
                raise ValueError(
                    "Serpent version {} must be one of "
                    "2.1.[29|30|31]".format(version)
                )
            self.options["results"]["serpentVersion"] = version

        useB1 = section.getboolean("useB1XS")

        if useB1 is not None:
            self.options["results"]["xs.getInfXS"] = False
            self.options["results"]["xs.getB1XS"] = True

        # TODO Document fission yield methodology
        # TODO Improve this to be more problem-generic
        fyEnergy = section.get("fission yield energy")
        if fyEnergy is not None:
            ene = self.fissionYieldEnergies.get(fyEnergy)
            if ene is not None:
                self.options["fission yield energy"] = ene
            else:
                raise ValueError(
                    "Fission yield energy must be {}, not {}".format(
                        ", ".join(sorted(self.fissionYieldEnergies)), fyEnergy
                    )
                )

    @requireBurnable
    def processDetectorFluxes(self, detectorfile, name):
        """Pull the universe fluxes from the detector file

        Does not perform any sorting on the tallies, so they must
        be loaded into the detector setting in an order that corresponds
        to the ordering expected by the rest of the framework.

        Parameters
        ----------
        detectorfile : str
            Path to the detector file to be read
        name : str
            Name of this specific detector to be read

        Returns
        -------
        numpy.ndarray
            Expected value of flux in each burnable universe

        """

        detector = self.read(detectorfile, "det")[name]
        tallies = detector.tallies

        if not detector.indexes:
            # Not uniquely binned quantities -> must be a single tallies quantity
            if tallies.size == 1:
                return tallies.reshape((1, 1))
            raise ValueError("Tally with bins but no indexes found")

        elif "universe" not in detector.indexes:
            raise ValueError(
                "Detector {} does not appear to be binned against "
                "universes. Indexes: {}".format(detector.name, detector.indexes)
            )

        if detector.indexes == ("universe",):
            if tallies.size != len(self.burnable):
                raise ValueError(
                    "Detector {} has {} tallies, expected {} for burnable "
                    "universes".format(detector.name, tallies.size, len(self.burnable))
                )

            return tallies.reshape((len(self.burnable), 1))

        elif "energy" in detector.indexes:
            if len(detector.indexes) != 2:
                raise ValueError(
                    "Detector {} must only be binned against universe, "
                    "and optionally energy, not {}".format(detector.indexes)
                )

            eneAx = detector.indexes.index("energy")
            uAx = detector.indexes.index("universe")

            if tallies.shape[uAx] != len(self.burnable):
                raise ValueError(
                    "Detector {} has {} tallies, expected {} for burnable "
                    "universes".format(detector.name, tallies.size, len(self.burnable))
                )

            return tallies.transpose((uAx, eneAx))

        raise ValueError(
            "Detector {} must only be binned against universe, "
            "and optionally energy, not {}".format(detector.indexes)
        )

    @requireBurnable
    def processMicroXS(self, mdepfile):

        microxs = self.read(mdepfile, "microxs").xsVal
        out = []

        for u in self.burnable:
            z = []
            r = []
            m = []

            for key, xs in microxs[u].items():
                z.append(key.zai)
                r.append(key.mt)
                # TODO Metastable
                m.append(xs)

            out.append(MicroXsVector.fromLongFormVectors(z, r, m, assumeSorted=False))

        return out

    @requireBurnable
    def processFissionYields(self):
        """Take fission yields for all isotopes"""
        # TODO Properly consider space, energy in fission yields
        # e.g. like how openmc.deplete provides a few options

        allYields = {}
        fyEne = self.options["fission yield energy"]

        for isotope in allIsotopes():
            if isotope.fissionYields is None:
                continue
            isoYields = isotope.fissionYields.get(fyEne)
            if isoYields is None:
                isoYields = isotope.fissionYields.at(0)
            allYields[isotope.zai] = isoYields
        return FakeSequence(allYields, len(self.burnable))
