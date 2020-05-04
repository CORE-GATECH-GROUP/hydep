import math
import logging
import numbers
from collections import defaultdict
import time

import numpy
from numpy.linalg import LinAlgError

from hydep import FailedSolverError
from hydep.constants import BARN_PER_CM2, EV_PER_JOULE, REACTION_MTS, FISSION_REACTIONS
from hydep.lib import ReducedOrderSolver
from hydep.internal import TransportResult
import hydep.internal.features as hdfeat
from .lib import applySFV, getAdjFwdEig
from .utils import NubarPolyFit

__all__ = ["SfvSolver"]


__logger__ = logging.getLogger("hydep.sfv")


class SfvSolver(ReducedOrderSolver):
    r"""Spatial flux variation (SFV) reduced order solver

    The solver has the following requirements:

    1. Fission matrix, with nodes in each burnable material
    2. Microscopic reaction cross sections, specifically absorption
        and fission cross sections
    3. Macroscopic absorption, nu sigma fission, and nubar cross
       sections in each burnable region

    .. note::

        Attributes :attr:`macroAbs0`, :attr:`macroNsf0`,
        :attr:`macroAbs1`, :attr:`macroFis1`, :attr:`kappaSigf1`,
        :attr:`extrapolatedNubar`, and :attr:`normalizedPhi0` will
        all be ``None`` until the first call to :meth:`processBOS`.
        These are exposed primarily for the sake of testing and
        should not be relied on.

    Configuration is done inside :meth:`beforeMain`.

    Attributes
    ----------
    numModes : int or None
        Positive number of modes used in the solution. A value of None indicates
        the simulation has not been configured
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
        reconstructed at the current substep. Value is meaningless
        until :meth:`substepUpdate`
    extrapolatedNubar : tuple of float or None
        Read-only view into homogenized :math:`\bar{\nu}` data
        projected to the current substep. Value is meaningless until
        :meth:`substepUpdate`.
    kappaFis1 : tuple of float or None
        Read-only view into product of node-homogenized energy
        deposited per fission :math:`\kappa` and :math:`\Sigma_f`
        [eV/cm]. Assumes all energy is deposited at fission site.
        Value is meaningless until :meth:`substepUpdate` and reflects
        the current substep.

    References
    ----------

    SFV : https://doi.org/10.1080/00295639.2019.1661171

    """
    _INDEX_XS_ABS_0 = 0
    _INDEX_XS_NSF_0 = 1
    _INDEX_XS_ABS_1 = 2
    _INDEX_XS_FIS_1 = 3  # NOTE Not nu-sigma fission
    _INDEX_VOL_K_FIS = 4  # NOTE node volumes * kappa * sigma f
    _INDEX_NUBAR = 5
    _INDEX_PHI_0 = 6
    _NUM_INDEXES = 7
    _NON_FISS_ABS_MT = frozenset(
        {REACTION_MTS.N_GAMMA, REACTION_MTS.N_2N, REACTION_MTS.N_3N}
    )

    def __init__(self):
        self._modes = None
        self._densityCutoff = None
        self._volumes = None
        self._totalvolume = None
        self._forwardMoments = None
        self._adjointMoments = None
        self._eigenvalues = None
        self._macroData = None
        self._keff0 = None
        self._currentPower = None
        self._isotopeFissionQs = None
        self._nubar = None

    @property
    def numModes(self):
        return self._modes

    @property
    def needs(self):
        return hdfeat.FeatureCollection(
            {hdfeat.FISSION_MATRIX, hdfeat.MICRO_REACTION_XS, hdfeat.HOMOG_LOCAL},
            {"abs", "nubar", "nsf"},
        )

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
        return tuple(self._macroData[:, self._INDEX_VOL_K_FIS] / self._volumes)

    def beforeMain(self, _model, manager, settings):
        """Prepare solver before main solution routines

        Parameters
        ----------
        _model : hydep.Model
            Geometry and materials for the problem to be solved.
            Currently not used, but provided to be consistent with
            the interface.
        manager : hydep.Manager
            Depletion interface
        settings : hydep.Settings
            Settings prescribed by the user. Will overwrite any
            currently stored values.

        """
        assert manager.burnable is not None
        vols = tuple(m.volume for m in manager.burnable)
        nvols = len(vols)

        sfvSettings = settings.sfv
        modes = sfvSettings.modes

        if modes is not None:
            if not isinstance(modes, numbers.Integral):
                raise TypeError(
                    f"Expected integer number of modes or None, got {type(modes)}"
                )
            elif not (0 < modes <= nvols):
                raise ValueError(
                    "Modes must be positive, and (for this problem) less than "
                    f"{nvols}, got {modes}"
                )
        else:
            modeFraction = sfvSettings.modeFraction
            if not isinstance(modeFraction, numbers.Real):
                raise TypeError(
                    f"Expected real for mode fraction, got {type(modeFraction)}"
                )
            elif not (0 < modeFraction <= 1):
                raise ValueError(
                    f"Mode fraction should be bounded (0, 1], got {modeFraction}"
                )
            modes = math.ceil(nvols * modeFraction)
            __logger__.info("Solving with %d modes", modes)
            __logger__.debug(
                "Computed from %d unique burnable materials with mode fraction %f",
                nvols,
                modeFraction,
            )
        self._modes = modes

        densityCutoff = sfvSettings._densityCutoff
        if not isinstance(densityCutoff, numbers.Real):
            raise TypeError(
                "Expected non-negative real for density cutoff, got "
                f"{type(densityCutoff)}"
            )
        elif densityCutoff < 0:
            raise ValueError(
                f"Expected non-negative real for density cutoff, got {densityCutoff}"
            )
        self._densityCutoff = densityCutoff

        # Nubar extrapolation
        fittingOrder = settings.fittingOrder
        if settings.unboundedFitting or settings.numFittingPoints is None:
            self._nubar = NubarPolyFit(order=fittingOrder, maxlen=None)
        elif fittingOrder >= settings.numFittingPoints:
            raise ValueError(
                f"Cannot make {fittingOrder} polynomial for nubar with "
                f"{settings.numFittingPoints} values"
            )
        else:
            self._nubar = NubarPolyFit(
                order=fittingOrder, maxlen=settings.numFittingPoints
            )

        self._volumes = vols
        self._totalvolume = sum(vols)
        self._macroData = numpy.empty((len(vols), self._NUM_INDEXES))
        self._processIsotopeFissionQ(manager.chain)

    def _processIsotopeFissionQ(self, isotopes):
        qvalues = defaultdict(float)
        for isotope in isotopes:
            for reaction in isotope.reactions:
                if reaction.mt in FISSION_REACTIONS:
                    qvalues[isotope.zai] += reaction.Q
        self._isotopeFissionQs = qvalues

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
        hydep.FailedSolverError
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
            raise FailedSolverError(f"{self!s} at {timestep}\n{le!s}") from le
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

    def substepSolve(self, timestep, compositions, microxs):
        """Apply the SFV method and predict the substep flux

        Parameters
        ----------
        timestep : hydep.internal.TimeStep
            Current point in calendar time
        compositions : hydep.internal.CompBundle
            Updated compositions in burnable materials
        microxs : iterable of hydep.internal.MicroXsVector
            Updated microscopic cross sections for this timestep

        Returns
        -------
        hydep.internal.TransportResult
            Transport result with the flux prediction and a (currently)
            meaningless multiplication factor

        """
        self._updateMacroFromMicroXs(compositions, microxs)
        self._macroData[:, self._INDEX_NUBAR] = self._nubar(timestep.currentTime)

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

        substepFlux = (
            normPrediction
            * self._currentPower
            / (normPrediction * self._macroData[:, self._INDEX_VOL_K_FIS]).sum()
        )

        end = time.time()

        return TransportResult(substepFlux, [numpy.nan, numpy.nan], runTime=end - start)

    def _updateMacroFromMicroXs(self, compositions, microxs):
        assert len(microxs) == self._macroData.shape[0]
        zais = tuple(iso.zai for iso in compositions.isotopes)
        cutoff = self._densityCutoff

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

                sigf = sum(isorxns.get(mt, 0) for mt in FISSION_REACTIONS)

                if sigf:
                    prod = sigf * comps[isox]
                    macroSigA += prod
                    macroSigF += prod
                    qSigmaF += prod * self._isotopeFissionQs[z]

            self._macroData[matix, self._INDEX_XS_ABS_1] = macroSigA
            self._macroData[matix, self._INDEX_XS_FIS_1] = macroSigF
            self._macroData[matix, self._INDEX_VOL_K_FIS] = qSigmaF

        self._macroData[:, (self._INDEX_XS_ABS_1, self._INDEX_XS_FIS_1)] *= BARN_PER_CM2

        self._macroData[:, self._INDEX_VOL_K_FIS] *= numpy.multiply(
            BARN_PER_CM2, self._volumes
        )
