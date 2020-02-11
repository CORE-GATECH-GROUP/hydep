import numbers
import warnings
from collections import defaultdict
import time

import numpy
from numpy.linalg import LinAlgError

from hydep.constants import BARN_PER_CM2, EV_PER_JOULE
from hydep.lib import ReducedOrderSolver
from hydep.internal import TransportResult
import hydep.internal.features as hdfeat
from .lib import applySFV, getAdjFwdEig
from .utils import NubarPolyFit

__all__ = ["SfvSolver"]


class SfvSolver(ReducedOrderSolver):
    r"""Spatial flux variation (SFV) reduced order solver

    Requirements
    ------------
    1. Fission matrix, with nodes in each burnable material
    2. Microscopic reaction cross sections, specifically absorption
        and fission cross sections
    3. Macroscopic absorption, nu sigma fission, and nubar cross
       sections in each burnable region

    .. note::

        Attributes :attr:`macroAbs0`, :attr:`macroNsf0`,
        :attr:`macroAbs1`, :attr:`macroFis1`,
        :attr:`extrapolatedNubar`, and :attr:`normalizedPhi0` will
        all be ``None`` until the first call to :meth:`processBOS`.
        These are exposed primarily for the sake of testing and
        should not be relied on.

    Parameters
    ----------
    numModes : int, optional
        Number of modes of the forward and adjoint flux to use. If not
        provided, :meth:`beforeMain` will set this value to be all
        possible modes
    densityCutoff : float, optional
        Threshold density [#/b-cm] that isotopes must exceed in order
        to be included in the cross section reconstruction
    numPreviousPoints : int, optional
        Number of previous points to use for extrapolations,
        specifically on :math:`\bar{\nu}`
    fittingOrder : int, optional
        Fitting order for polynomials like :math:`\bar{\nu}`

    Attributes
    ----------
    numModes : int or None
        Number of modes of the forward and adjoint flux that are used
    densityCutoff : float
        Cutoff density [#/b/cm] used in cross section reconstruction
    needs : hydep.internal.features.FeatureCollection
        Physics required by this solver and macroscopic cross sections.
    macroAbs0 : tuple of float or None
        Read-only view into beginning-of-step macroscopic absorption
        cross sections stored on the solver.
    macroNsf0 : tuple of float or None
        Read-only view into beginning-of-step :math:`\bar{\nu}\Sigma_f`
        data stored on the solver
    normalizedPhi0 : tuple of float or None
        Read-only view into beginning-of-step fluxes, normalized such
        that total power production is unity
    macroAbs1 : tuple of float or None
        Read-only view into macroscopic absorption reconstructed at the
        substep. Value is meaningless until :meth:`substepUpdate`
    macroFis1 : tuple of float or None
        Read-only view into macroscopic fission cross sections
        econstructed at the current substep. Value is meaningless
        until :meth:`substepUpdate`
    extrapolatedNubar : tuple of float or None
        Read-only view into homogenized :math:`\bar{\nu}` data
        projected to the current substep. Value is meaningless until
        :meth:`substepUpdate`.
    kappaFis1 : tuple of float or None
        Read-only view into energy-production per fission, assuming all
        energy is deposited at fission site. Value is meaningless until
        :meth:`substepUpdate` and reflects the current substep.

    References
    ----------

    SFV : https://doi.org/10.1080/00295639.2019.1661171

    """
    _INDEX_XS_ABS_0 = 0
    _INDEX_XS_NSF_0 = 1
    _INDEX_XS_ABS_1 = 2
    _INDEX_XS_FIS_1 = 3  # NOTE Not nu-sigma fission
    _INDEX_XS_Q_FIS = 4
    _INDEX_NUBAR = 5
    _INDEX_PHI_0 = 6
    _INDEX_PHI_1 = 7
    _NUM_INDEXES = 8
    _FIS_MT = 18
    _NON_FISS_ABS_MT = {102, 16, 17}

    def __init__(
        self, numModes=None, densityCutoff=0, numPreviousPoints=3, fittingOrder=1,
    ):
        self._volumes = None
        if numModes is None:
            self._numModes = None
        else:
            self.numModes = numModes
        self.densityCutoff = densityCutoff

        self._totalvolume = None
        self._forwardMoments = None
        self._adjointMoments = None
        self._eigenvalues = None
        self._macroData = None
        self._keff0 = None
        self._currentPower = None
        self._isotopeFissionQs = None
        self._nubar = NubarPolyFit(maxlen=numPreviousPoints, order=fittingOrder)

    @property
    def needs(self):
        return hdfeat.FeatureCollection(
            {hdfeat.FISSION_MATRIX, hdfeat.MICRO_REACTION_XS, hdfeat.HOMOG_LOCAL},
            {"abs", "nubar", "nsf"},
        )

    @property
    def numModes(self):
        return self._numModes

    @numModes.setter
    def numModes(self, value):
        if not isinstance(value, numbers.Integral):
            raise TypeError(f"Modes must be integral, not {type(value)}")
        elif value <= 0:
            raise ValueError(f"Modes must be positive, not {value}")
        elif self._volumes is not None and value > len(self._volumes):
            raise ValueError(
                f"More modes requested than allowable: {value} vs. "
                "{len(self._volumes)}"
            )
        self._numModes = value

    @property
    def densityCutoff(self):
        return self._densityCutoff

    @densityCutoff.setter
    def densityCutoff(self, value):
        if not isinstance(value, numbers.Real):
            raise TypeError(value)
        elif value < 0:
            raise ValueError(value)
        self._densityCutoff = value

    @property
    def macroAbs0(self):
        if self._macroData is None:
            return None
        return tuple(self._macroData[:, self._INDEX_XS_ABS_0])

    @property
    def macroNsf0(self):
        if self._macroData is None:
            return None
        return tuple(self._macroData[:, self._INDEX_XS_NSF_0])

    @property
    def macroAbs1(self):
        if self._macroData is None:
            return None
        return tuple(self._macroData[:, self._INDEX_XS_ABS_1])

    @property
    def macroFis1(self):
        if self._macroData is None:
            return None
        return tuple(self._macroData[:, self._INDEX_XS_FIS_1])

    @property
    def extrapolatedNubar(self):
        if self._macroData is None:
            return None
        return tuple(self._macroData[:, self._INDEX_NUBAR])

    @property
    def normalizedPhi0(self):
        if self._macroData is None:
            return None
        return tuple(self._macroData[:, self._INDEX_PHI_0])

    @property
    def kappaSigf1(self):
        if self._macroData is None:
            return None
        return tuple(self._macroData[:, self._INDEX_XS_Q_FIS])

    def beforeMain(self, _model, burnedmats):
        """Prepare solver before main solution routines

        If :attr:`numModes` is not set, then it will be set to the
        number of burnable materials. If more modes are requested than
        burnable materials, a warning is raised and the value is
        lowered to the number of burnable materials.

        Parameters
        ----------
        _model : hydep.Model
            Geometry and materials for the problem to be solved.
            Currently not used, but provided to be consistent with
            the interface.
        burnedmats : iterable of hydep.BurnableMaterials
            Burnable materials, ordered to be consistent with the
            other solvers

        """
        vols = tuple(m.volume for m in burnedmats)
        nvols = len(vols)

        if self.numModes is None:
            warnings.warn(
                f"No self.numModes selected for {self!s}, using {nvols}",
                RuntimeWarning,
            )
            self.numModes = nvols
        elif self.numModes > nvols:
            warnings.warn(
                f"Cannot extract {self.numModes} from {nvols} materials, using "
                f"{nvols}",
                RuntimeWarning,
            )
            self.numModes = nvols
        self._volumes = vols
        self._totalvolume = sum(vols)
        self._macroData = numpy.empty((len(vols), self._NUM_INDEXES))

    def processBOS(self, txresult, timestep, power):
        """Process data from the beginning of step high fidelity solution

        Parameters
        ----------
        txResult : hydep.internal.TransportResult
            Transport result from the :class:`HighFidelitySolver`
        timestep : hydep.internal.TimeStep
            Representation of the point in calendar time, and
            step in simulation.
        power : float
            Current system power [W]. To be held constant across the
            substep interval.

        Raises
        ------
        AttributeError
            If no fission matrix is supplied and no fission matrix data
            is current stored.
        numpy.linalg.LinAlgError
            If the eigenvalue solution of the fission matrix failed

        """
        if txresult.fmtx is None:
            if self._forwardMoments is None:
                raise AttributeError(
                    f"No fission matrix currently nor previously passed at {timestep}"
                )
        else:
            self._bosProcessFmtx(txresult, timestep)
        self._bosProcessMacroXs(txresult.macroXS)
        self._keff0 = txresult.keff[0]
        self._bosProcessFlux(txresult.flux)
        self._currentPower = power * EV_PER_JOULE
        nubar = [m["nubar"] for m in txresult.macroXS]
        self._nubar.insort(timestep.currentTime, nubar)

    def _bosProcessFmtx(self, txresult, timestep):
        try:
            adj, fwd, eig = getAdjFwdEig(txresult.fmtx)
        except LinAlgError as le:
            raise LinAlgError(f"{self!s} at {timestep}\n{le!s}")
        self._adjointMoments = adj[:, : self.numModes]
        self._forwardMoments = fwd[:, : self.numModes]
        self._eigenvalues = eig[: self.numModes]

    def _bosProcessMacroXs(self, macroxs):
        assert len(macroxs) == len(self._volumes)
        # Only works with one-group
        for ix, matdata in enumerate(macroxs):
            self._macroData[ix, self._INDEX_XS_ABS_0] = matdata["abs"][0]
            self._macroData[ix, self._INDEX_XS_NSF_0] = matdata["nsf"][0]

    def _bosProcessFlux(self, flux):
        assert len(flux) == len(self._volumes)
        assert len(flux.shape) == 2
        if flux.shape[1] != 1:
            raise NotImplementedError(f"Mutligroup flux not supported with {self!s}")
        self._macroData[:, self._INDEX_PHI_0] = (
            flux[:, 0] * self._totalvolume / (flux[:, 0] * self._volumes).sum()
        )

    def substepUpdate(self, timestep, compositions, microxs):
        """Prepare the solver for the execution stage

        Parameters
        ----------
        timestep : hydep.internal.TimeStep
            Current point in calendar time
        compositions : hydep.internal.CompBundle
            Updated compositions in burnable materials
        microxs : iterable of hydep.internal.MicroXsVector
            Updated microscopic cross sections for this timestep

        """
        if self._isotopeFissionQs is None:
            self._processIsotopeFissionQ(compositions.isotopes)
        self._updateMacroFromMicroXs(compositions, microxs)
        self._macroData[:, self._INDEX_NUBAR] = self._nubar(timestep.currentTime)

    def _processIsotopeFissionQ(self, isotopes):
        qvalues = defaultdict(float)
        for isotope in isotopes:
            for reaction in isotope.reactions:
                if reaction.mt == self._FIS_MT:
                    qvalues[isotope.zai] = reaction.Q
                    break
        self._isotopeFissionQs = qvalues

    def _updateMacroFromMicroXs(self, compositions, microxs):
        assert len(microxs) == self._macroData.shape[1]
        zais = tuple(iso.zai for iso in compositions.isotopes)
        cutoff = self.densityCutoff

        # TODO Subprocess??
        for matix, (comps, matxs) in enumerate(zip(compositions.densities, microxs)):
            macroSigA = 0
            macroSigF = 0
            qSigmaF = 0
            for isox, z in enumerate(zais):
                if comps[isox] < cutoff:
                    continue
                isorxns = matxs.getReactions(z)
                if isorxns is None:
                    continue

                siga = 0
                for mt in self._NON_FISS_ABS_MT:
                    siga += isorxns.get(mt, 0.0)
                macroSigA += siga * comps[isox]

                sigf = isorxns.get(self._FIS_MT)

                if sigf is not None:
                    prod = sigf * comps[isox]
                    macroSigA += prod
                    macroSigF += prod
                    qSigmaF += prod * self._isotopeFissionQs[z]

            self._macroData[matix, self._INDEX_XS_ABS_1] = macroSigA
            self._macroData[matix, self._INDEX_XS_FIS_1] = macroSigF
            self._macroData[matix, self._INDEX_XS_Q_FIS] = qSigmaF

        self._macroData[:, (self._INDEX_XS_ABS_1, self._INDEX_XS_FIS_1)] *= BARN_PER_CM2

        self._macroData[:, self._INDEX_XS_Q_FIS] *= numpy.multiply(
            BARN_PER_CM2, self._volumes
        )

    def execute(self) -> float:
        """Perform the prediction and store the new flux internally

        Returns
        -------
        float
            Time required to perform the prediction and re-normalize
            fluxes to appropriate levels

        """
        start = time.time()
        data = self._macroData

        normPrediction = applySFV(
            data[:, self._INDEX_XS_ABS_0],
            data[:, self._INDEX_XS_ABS_1],
            data[:, self._INDEX_XS_NSF_0],
            data[:, self._INDEX_XS_FIS_1] * data[:, self._INDEX_NUBAR],
            self._keff0,
            self._adjointMoments,
            self._forwardMoments,
            self._eigenvalues,
            data[:, self._INDEX_PHI_0],
            overwrite=False,
        )

        data[:, self._INDEX_PHI_1] = (
            normPrediction
            * self._currentPower
            / (normPrediction * self._macroData[:, self._INDEX_XS_Q_FIS]).sum()
        )

        return time.time() - start

    def processResults(self) -> TransportResult:
        """Process the new fluxes"""
        # TODO Maybe have execute do **everything**?
        return TransportResult(
            self._macroData[:, self._INDEX_PHI_1].copy(), [numpy.nan, numpy.nan]
        )
