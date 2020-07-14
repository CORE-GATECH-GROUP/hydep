"""
Class responsible for processing Serpent outputs
"""
import warnings
import copy
from functools import wraps
import textwrap
import math
import collections
import typing

import numpy
import serpentTools

from hydep.internal import MaterialDataArray, XsIndex
from hydep.constants import CM2_PER_BARN, REACTION_MTS
from .fmtx import parseFmtx


__all__ = ["SerpentProcessor", "FissionYieldFetcher"]


ResTuple = collections.namedtuple("ResTuple", "keff macroXS")
ResTuple.__doc__ = """Small bundle for result file data

Parameters
----------
keff : numpy.ndarray
    Array with multiplication factor and absolute uncertainty
macroXS : list of dict or None
    Macroscopic cross sections in every burnable region.
    Able to be passed directly to
    :attr:`hydep.internal.TransportResult.macroXS`

"""


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
    burnable : tuple of str, optional
        Names of universes with burnable materials. If not given,
        must be provided prior to processing files
    reactionIndex : hydep.internal.XsIndex, optional
        Index to be used when fetching microscopic cross sections

    Attributes
    ----------
    burnable : tuple of str or None
        Universes that contain burnable materials. Will attempt to
        pull data from these universes, including group fluxes,
        macroscopic cross sections, microscopic cross sections, and
        properties of the fission matrix. Must be set prior to
        processing files.
    reactionIndex : hydep.internal.XsIndex or None
        Index to be used when fetching microscopic cross sections.
        Must be set prior to :meth:`processMicroXS`
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

    def __init__(self, burnable=None, reactionIndex=None):
        self._burnable = burnable
        self.fyHelper = None
        self.reactionIndex = reactionIndex

    @property
    def burnable(self):
        return self._burnable

    @burnable.setter
    def burnable(self, b):
        if b is None:
            self._burnable = None
        else:
            self._burnable = tuple(b)

    @property
    def reactionIndex(self):
        return self._reactionIndex

    @reactionIndex.setter
    def reactionIndex(self, ix):
        if ix is None or isinstance(ix, XsIndex):
            self._reactionIndex = ix
        else:
            raise TypeError(
                f"Reaction index must be None or XsIndex, not {type(ix)}"
            )

    @staticmethod
    def _warnOptions(settings, reason):
        pairs = ("{}: {}".format(k, v) for k, v in settings.items())
        msg = (
            "The following settings could not be used to control "
            "Serpent reading:\n{}\nReason: {}".format(
                "\n  ".join(pairs), reason
            )
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
        opts = self.options.get(filetype)
        if opts is None:
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

    def getKeff(self, resultfile, index=0) -> typing.Tuple[float, float]:
        """Pull just the multiplication factor from a result file

        Parameters
        ----------
        resultfile : str
            Path to the result file to be read
        index : int, optional
            Time step

        Returns
        -------
        Iterable[float, float]
            Multiplication factor and absolute uncertainty

        """
        results = self.read(resultfile, "results")

        keff = results.resdata["absKeff"]
        if len(keff.shape) == 2:
            keff = keff[index]
        keff[1] *= keff[0]
        return keff

    @requireBurnable
    def processResult(self, resultfile, reqXS, index=0) -> ResTuple:
        """Scrape multiplication factor and xs

        Expects all the universes to have been selected for
        homogenization with the ``set gcu`` card. If this is not the
        case, there will be errors. Examine some of the smaller methods
        that are used here to pull a single quantity

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
        index : int, optional
            Time step from which to pull all data

        Returns
        -------
        ResTuple

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
          region from a detector file

        """

        results = self.read(resultfile, "results")

        keff = results.resdata["absKeff"]
        if len(keff.shape) == 2:
            keff = keff[index]
        keff[1] *= keff[0]

        xsLeader = "inf" if self.options["results"]["xs.getInfXS"] else "b1"
        allXS = []

        for univKey in self.burnable:
            xsdata = {}
            universe = results.getUniv(univKey, index=index)
            source = getattr(universe, xsLeader + "Exp")
            for reqKey in reqXS:
                xsdata[reqKey] = source[xsLeader + reqKey.capitalize()]
            allXS.append(xsdata)

        return ResTuple(keff, allXS)

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
        # Would like to share the reading with the processFissionYields
        # method in the future
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
                "universes. Indexes: {}".format(
                    detector.name, detector.indexes
                )
            )

        if detector.indexes == ("universe",):
            if tallies.size != len(self.burnable):
                raise ValueError(
                    "Detector {} has {} tallies, expected {} for burnable "
                    "universes".format(
                        detector.name, tallies.size, len(self.burnable)
                    )
                )

            return tallies.reshape((len(self.burnable), 1))

        elif "energy" in detector.indexes:
            if len(detector.indexes) != 2:
                raise ValueError(
                    "Detector {} must only be binned against universe, "
                    "and optionally energy, not {}".format(
                        detector.name, detector.indexes
                    )
                )

            eneAx = detector.indexes.index("energy")
            uAx = detector.indexes.index("universe")

            if tallies.shape[uAx] != len(self.burnable):
                raise ValueError(
                    "Detector {} has {} tallies, expected {} for burnable "
                    "universes".format(
                        detector.name, tallies.size, len(self.burnable)
                    )
                )

            return tallies.transpose((uAx, eneAx))

        raise ValueError(
            "Detector {} must only be binned against universe, "
            "and optionally energy, not {}".format(
                detector.name, detector.indexes
            )
        )

    @requireBurnable
    def processMicroXS(self, mdepfile) -> MaterialDataArray:
        if self.reactionIndex is None:
            raise AttributeError("Reaction index for {self} not set")

        microxs = self.read(mdepfile, "microxs").xsVal

        data = numpy.empty(
            (len(self.burnable), len(self.reactionIndex)), dtype=numpy.float64,
        )

        for uindex, univ in enumerate(self.burnable):
            univxs = microxs[univ]
            for rxnIndex, (zai, rxn) in enumerate(self.reactionIndex):
                # Keys to microxs are (zai, rxn, metastable), where
                # metastable indicates if the reaction goes to a ground
                # or metastable state. These are handled by branching ratios
                # on the chain
                data[uindex, rxnIndex] = univxs.get((zai, rxn, 0), 0.0)

        return MaterialDataArray(self.reactionIndex, data * CM2_PER_BARN)

    @requireBurnable
    def processFissionYields(self, detectorfile):
        """Take fission yields for all isotopes"""
        assert self.fyHelper is not None
        fydet = self.read(detectorfile, "det")
        return self.fyHelper.collapseYieldsFromDetectors(
            fydet.detectors.values()
        )


class FissionYieldFetcher:
    """Helper for getting energy-averaged fission yields from Serpent

    Inspired by :class:`openmc.deplete.helpers.AveragedFissionYieldHelper`

    Parameters
    ----------
    matids : iterable of str
        Ordering of universes to be consistent with the remainder
        of the framework. Fission rates will be tallied in these
        universes.
    isotopes : iterable of hydep.internal.Isotope
        Isotopes that may or may not have fission yields. Will
        skip isotopes without yields.
    upperEnergy : float, optional
        Maximum energy [MeV] above which fission events will not
        contribute to the weighting

    """

    COMPONENT_FISSIONS = {
        942400: {
            REACTION_MTS.FIRST_CHANCE_FISSION,
            REACTION_MTS.SECOND_CHANCE_FISSION,
            REACTION_MTS.THIRD_CHANCE_FISSION,
        },
        922340: {
            REACTION_MTS.FIRST_CHANCE_FISSION,
            REACTION_MTS.SECOND_CHANCE_FISSION,
            REACTION_MTS.THIRD_CHANCE_FISSION,
        },
        922360: {
            REACTION_MTS.FIRST_CHANCE_FISSION,
            REACTION_MTS.SECOND_CHANCE_FISSION,
            REACTION_MTS.THIRD_CHANCE_FISSION,
        },
        962430: {
            REACTION_MTS.FIRST_CHANCE_FISSION,
            REACTION_MTS.SECOND_CHANCE_FISSION,
        },
    }
    # These isotopes don't have a total fission (MT=18)
    # reaction in my ENDF/B-VII.0 library that shipped
    # with Serpent. All others will use a single total
    # fission reaction
    FISSION_MT = REACTION_MTS.TOTAL_FISSION

    def __init__(self, matids, isotopes, upperEnergy=20):
        self._constant = {}
        self._variable = {}
        self._ucards = textwrap.fill(
            " ".join(["du {}".format(u) for u in matids])
        )
        for iso in isotopes:
            if iso.fissionYields is None:
                continue
            if len(iso.fissionYields.energies) == 1:
                self._constant[iso.zai] = iso.fissionYields.at(0)
            else:
                self._variable[iso.zai] = iso.fissionYields
        self.upperEnergy = upperEnergy

    def makeDetectors(self) -> list:
        """Produce lines that can be used to write detector inputs

        Returns
        -------
        list of str
            Lines that can be written to a Serpent input file
            to construct detectors

        """
        materials = []
        detectors = []
        for zai, fy in self._variable.items():
            materials.append(f"mat fy{zai} 1.0 {int(zai/10)}.09c 1")
            # Not assuming metastables are fissionable and thus
            # don't need to be remapped to different names
            # Also assuming 900K for all fuels
            gridname = f"fyenergies{zai}"

            energies = []
            for ix, lower in enumerate(fy.energies[:-1]):
                lethargyMid = math.sqrt(lower * fy.energies[ix + 1]) / 1e6
                energies.append(f"{lethargyMid:.3E}")

            detectors.append(
                f"ene {gridname} 1 0.0 {' '.join(energies)} {self.upperEnergy}"
            )
            rxns = " ".join(
                [
                    f"dr {r} fy{zai}"
                    for r in self.COMPONENT_FISSIONS.get(
                        zai, {REACTION_MTS.TOTAL_FISSION}
                    )
                ]
            )
            detectors.append(f"det fy{zai} de {gridname} {rxns}")
            detectors.append(self._ucards)
        return materials + detectors

    def collapseYieldsFromDetectors(
        self, detectors
    ) -> typing.List[typing.Dict[int, "hydep.internal.FissionYield"]]:
        """Obtain region specific, fission-rate-averaged fission yields

        Parameters
        ----------
        detectors : iterable of :class:`serpentTools.Detector`
            Detectors read from the detector file that were created
            with :meth:`makeDetectors`

        Returns
        -------
        list of dict
            List of fission yield dictionaries where ``l[ix]`` maps
            parent ZAI to :class:`hydep.internal.FissionYield`
            for region ``ix``

        """
        materialYields = []
        for d in detectors:
            if not d.name.startswith("fy"):
                continue
            zai = int(d.name[2:])
            weights = self._getweights(d)
            assert len(weights.shape) == 2
            colYields = self._collapseIsoYields(weights, self._variable[zai])

            if not materialYields:
                for slab in colYields:
                    matweights = self._constant.copy()
                    matweights[zai] = slab
                    materialYields.append(matweights)
            else:
                for ix, slab in enumerate(colYields):
                    materialYields[ix][zai] = slab

        return materialYields

    @staticmethod
    def _getweights(d):
        # Detectors will come in with binned
        # (energy, universe, reaction)
        # If only one universe, the universe axis will be removed
        # and the string "universe" will not be in the indexes attribute
        # If one reaction, the string "reaction" will not be in the
        # indexes, and the reaction index will be removed
        # Arrays need to leave here of shape (universes, energy)
        # where universes >= 1, energy >= 1
        assert "energy" in d.indexes
        if len(d.indexes) == 1:
            return (d.tallies / d.tallies.sum())[numpy.newaxis]

        # Collapse the isotopes w/o a single fission 18 reaction
        if "reaction" in d.indexes:
            tallies = d.tallies.sum(d.indexes.index("reaction"))
            # Full case
            if "universe" in d.indexes:
                tallies = tallies.transpose()
            else:
                tallies = tallies[numpy.newaxis]
        elif "universe" in d.indexes:
            tallies = d.tallies.transpose()
        return tallies / tallies.sum(axis=1, keepdims=True)

    @staticmethod
    def _collapseIsoYields(weights, eneyields):
        matyields = []
        # Order must be fy * weight or else we get subject to
        # numpy dispatching and return weight of just product
        # zais
        for matweights in weights:
            fy = eneyields.at(0) * matweights[0]
            for ix, w in enumerate(matweights[1:], start=1):
                fy += eneyields.at(ix) * w
            matyields.append(fy)
        return matyields
