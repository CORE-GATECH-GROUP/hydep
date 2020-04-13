"""
Depletion manager

Control time-steps, material divisions, depletion, etc
"""

import warnings
import numbers
from collections.abc import Sequence, Callable
from itertools import repeat, starmap
import multiprocessing

import numpy

from hydep import (
    BurnableMaterial,
    DepletionChain,
    NegativeDensityWarning,
    NegativeDensityError,
)
from hydep.constants import SECONDS_PER_DAY
from hydep.typed import TypedAttr, IterableOf
from hydep.internal import Cram16Solver, Cram48Solver, CompBundle
from hydep.internal.features import FeatureCollection, MICRO_REACTION_XS, FISSION_YIELDS
from hydep.internal.utils import FakeSequence


__all__ = ["Manager"]


class Manager:
    """Primary depletion manager

    Responsible for depleting materials and updating compositions

    Parameters
    ----------
    chain : hydep.DepletionChain
        Chain describing how isotopes decay and transmute
    daysteps : iterable of float
        Length in time [d] for each coarse depletion step
    power : float or iterable of float
        Power [W] to deplete the entire problem. If a single value
        is given, a constant power will be used. Otherwise, must
        have the same number of elements as ``daysteps``, corresponding
        to a constant power in each depletion step
    substepDivision : int or sequence of int
        Number of substeps to divide each entry in ``daysteps``, also
        the number of transport solutions, high fidelity and reduced
        order, per entry. If an integer, the value will be applied to
        all entries.
    numPreliminary : int, optional
        Number of coarse depletion steps to take before engaging in
        coupled behavior. Useful for approaching some equilibrium value
        with smaller steps with the high fidelity code
    depletionSolver : string or int or callable, optional
        Value to use in configuring the depletion solver. Passed to
        :meth:`setDepletionSolver`
    negativeDensityWarnPercent: float, optional
        Threshold for warning about negative densities. Treated
        as a percentage of positive densities, range [0, 1]. Defaults
        to 0.01 %
    negativeDensityErrorPercent : float, optional
        Threshold for raising an error on negative densities. Treated
        as a percentage of positive densities, range [0, 1]. Defaults to 1.

    Attributes
    ----------
    chain : hydep.DepletionChain
        Depletion chain
    timesteps : numpy.ndarray
        Length of time [s] for each coarse depletion step
    powers : numpy.ndarray
        Power [W] to use for each coarse depletion steps
    numPreliminary : int
        Number of coarse depletion steps to take before engaging in
        coupled behavior. Useful for approaching some equilibrium value
        with smaller steps with the high fidelity code
    burnable : None or tuple of hydep.BurnableMaterial
        Ordering of burnable materials. Must be set prior to depletion
        to maintain a consistent mapping from reaction rates and
        compositions
    needs : hydep.internal.features.FeatureCollection
        Read-only property describing the capabilities any solver must
        have in order to properly perform the depletion analysis.
        Currently requires calculation of isotopic reaction cross
        sections in each burnable material, as well as the flux.
    substeps : sequence of int
        Number of transport solutions, high fidelity and reduced order,
        per coarse depletion step. Not writable.
    negativeDensityWarnPercent : float
        Percentage threshold for warning about negative densities,
        range [0, 1]. Must be less than :attr:`negativeDensityErrorPercent`
    negativeDensityErrorPercent : float
        Percentage threshold for raising and error on negative
        densities, range [0, 1]. Must be greater than
        :attr:`negativeDensityWarnPercent`

    """

    chain = TypedAttr("chain", DepletionChain)
    _burnable = IterableOf("burnable", BurnableMaterial, allowNone=True)

    def __init__(
        self,
        chain,
        daysteps,
        power,
        substepDivision,
        numPreliminary=0,
        depletionSolver=None,
        negativeDensityWarnPercent=1E-4,
        negativeDensityErrorPercent=1,
    ):
        self.chain = chain

        daysteps = numpy.asarray(daysteps, dtype=float)
        if len(daysteps.shape) > 1:
            raise TypeError("Day steps must be vector, not array")
        self.timesteps = tuple(daysteps * SECONDS_PER_DAY)

        self.powers = tuple(self._validatePowers(power))

        self._burnable = None

        if numPreliminary is None:
            self._nprelim = 0
        else:
            if not isinstance(numPreliminary, numbers.Integral):
                raise TypeError(
                    f"Non-integer preliminary steps not allowed: {type(numPreliminary)}"
                )
            elif not (0 <= numPreliminary < len(self.timesteps)):
                raise ValueError(
                    "Number of preliminary steps must be between [0, "
                    f"{len(self.timesteps)}), not {numPreliminary}"
                )
            self._nprelim = numPreliminary

        self._substeps = self._validateSubsteps(substepDivision)

        self.setDepletionSolver(depletionSolver)

        self._negativeDensityWarn = 0
        self._negativeDensityError = 1
        self.negativeDensityWarnPercent = negativeDensityWarnPercent
        self.negativeDensityErrorPercent = negativeDensityErrorPercent

    def _validatePowers(self, power):
        if isinstance(power, numbers.Real):
            if power <= 0:
                raise ValueError(f"Power must be positive, not {power}")
            return repeat(power, len(self.timesteps))
        elif isinstance(power, Sequence):
            if len(power) != len(self.timesteps):
                raise ValueError(
                    f"Number of powers {len(power)} differ from steps "
                    f"{len(self.timesteps)}"
                )
            for p in power:
                if not isinstance(p, numbers.Real):
                    raise TypeError(
                        "Power must be positive real, or vector of positive real. "
                        f"Found {p}"
                    )
                elif p <= 0:
                    raise ValueError(
                        "Power must be positive real, or vector of positive real. "
                        f"Found {p}"
                    )

            return power
        else:
            raise TypeError(
                "Power must be positive real, or vector of positive real, "
                f"not {type(power)}"
            )

    def _validateSubsteps(self, divisions):
        maxallowed = len(self.timesteps) - self._nprelim
        if isinstance(divisions, numbers.Integral):
            if divisions <= 0:
                raise ValueError(f"Divisions must be positive integer, not {divisions}")
            return FakeSequence(divisions, maxallowed)
        if isinstance(divisions, (numpy.ndarray, Sequence)):
            if len(divisions) != maxallowed:
                raise ValueError(
                    "Number of divisions {} not equal to number of time "
                    "steps {} - number of preliminary steps {}".format(
                        len(divisions), len(self.timesteps), self._nprelim
                    )
                )
            substeps = []
            for value in divisions:
                if not isinstance(value, numbers.Integral):
                    raise TypeError(
                        f"Divisions must be positive integer, not {type(value)}"
                    )
                elif value <= 0:
                    raise ValueError(f"Divisions must be positive integer, not {value}")
                substeps.append(value)
            return tuple(substeps)

        raise TypeError(
            "Substeps must be postive integer, or sequence of positive "
            f"integer, not {divisions}"
        )

    def setDepletionSolver(self, solver):
        """Configure the depletion solver

        Solver can either be a string, e.g. ``"cram16"``,
        integer, ``16``, or a callable function. Callable functions
        should fulfill the following requirements:

        1. Be importable / pickle-able in order to be dispatched via
           :meth:`multiprocessing.Pool.starmap`
        2. Have a call signature ``solver(A, N0, dt)`` where ``A`` is
           the :class:`scipy.sparse.csr_matrix` sparse representation
           of the depletion matrix with shape ``N x N``, ``N0`` is a
           :class:`numpy.ndarray` with the beginning-of-step
           compositions, and ``dt`` is the length of the depletion
           interval in seconds

        For the time being, no introspection is performed to ensure
        that the correct signature is used. String values are
        case-insensitive, and integers indicate the order of CRAM
        to be used.

        Parameters
        ----------
        solver : str or int or callable or None
            Item indicating what solver should be used. A value
            of ``None`` reverts to the default CRAM16.

        Raises
        ------
        TypeError
            If ``solver`` doesn't match any requirements

        """

        if solver is None:
            self._depsolver = Cram16Solver.__call__
            return

        if isinstance(solver, str):
            solver = solver.lower()

        candidate = {
            "cram16": Cram16Solver,
            "cram48": Cram48Solver,
            16: Cram16Solver,
            48: Cram48Solver,
            "16": Cram16Solver,
            "48": Cram48Solver,
        }.get(solver)

        if candidate is not None:
            self._depsolver = candidate.__call__
            return

        if isinstance(solver, Callable):
            self._depsolver = solver
            return solver

        raise TypeError(f"Could not decipher {solver} of type {type(solver)}")

    @property
    def burnable(self):
        return self._burnable

    @property
    def needs(self):
        return FeatureCollection({MICRO_REACTION_XS, FISSION_YIELDS})

    @property
    def numPreliminary(self):
        return self._nprelim

    @property
    def substeps(self):
        return self._substeps

    @property
    def negativeDensityWarnPercent(self):
        return self._negativeDensityWarn

    @negativeDensityWarnPercent.setter
    def negativeDensityWarnPercent(self, value):
        if not isinstance(value, numbers.Real):
            raise TypeError(
                f"Negative density threshold must be real, not {type(value)}"
            )
        if not 0 <= value <= self._negativeDensityError:
            raise ValueError(
                "Negative density warning must be between 0 (inclusive) and "
                f"{self._negativeDensityError:.5f} [error threshold] "
                f"not {value}"
            )
        self._negativeDensityWarn = value

    @property
    def negativeDensityErrorPercent(self):
        return self._negativeDensityError

    @negativeDensityErrorPercent.setter
    def negativeDensityErrorPercent(self, value):
        if not isinstance(value, numbers.Real):
            raise TypeError(
                f"Negative density threshold must be real, not {type(value)}"
            )
        if not self.negativeDensityWarnPercent <= value <= 1:
            raise ValueError(
                "Negative density warning must be between "
                f"{self.negativeDensityWarnPercent:.5f} [warning threshold] "
                f"(inclusive) and 1 (inclusive), not {value}"
            )
        self._negativeDensityError = value

    def preliminarySteps(self):
        """Iterate over preliminary time steps and powers

        Useful for running only a high fidelity solver before
        substep depletion with a reduced order solver

        Yields
        ------
        float
            Time step [s]
        float
            Power [W] for current step
        """
        return zip(self.timesteps[: self._nprelim], self.powers[: self._nprelim])

    def activeSteps(self):
        """Iterate over active time steps and powers

        These are steps after the :meth:`preliminarySteps`.

        Yields
        ------
        float
            Time step [s]
        float
            Power [W] for current step
        """
        return zip(self.timesteps[self._nprelim :], self.powers[self._nprelim :])

    def beforeMain(self, model, settings=None):
        """Check that all materials have volumes and set indexes

        Parameters
        ----------
        model : hydep.Model
            Problem to be solved. Must contain at least one
            :class:`hydep.BurnableMaterial`
        settings : hydep.settings.HydepSettings, optional
            Settings for the framework. Use to configure depletion
            solver.

        """
        if settings is not None:
            solver = settings.depletionSolver
            if solver is not None:
                self.setDepletionSolver(solver)

        burnable = tuple(model.root.findBurnableMaterials())

        if not burnable:
            raise ValueError(f"No burnable materials found in {model}")

        for ix, mat in enumerate(burnable):
            if mat.volume is None:
                raise AttributeError(
                    f"{mat.__class__} {mat.name} does not have a volume set"
                )
            mat.index = ix

        self._burnable = burnable

    def deplete(self, dtSeconds, concentrations, reactionRates, fissionYields):
        """Deplete all burnable materials

        Parameters
        ----------
        dtSeconds : float
            Length of depletion interval in seconds
        concentrations : hydep.internal.CompBundle
            Incoming material compositions. The
            :attr:`hydep.internal.CompBundle.isotopes` will be
            consistent between the incoming and outgoing bundle
        reactionRates : iterable of hydep.internal.MicroXsVector
            Stand in for microscopic reaction rates in each burnable
            material
        fissionYields : iterable of hydep.internal.FissionYield
            Fission yields in each burnable material

        Returns
        -------
        hydep.internal.CompBundle
            New compositions for each burnable material and the isotope
            ordering. Densities will be non-negative

        Raises
        ------
        hydep.NegativeDensityError
            If the sum of any and all negative densites computed exceeded
            the :attr:`negativeDensityErrorPercent`

        Warns
        -----
        hydep.NegativeDensityWarning
            If the sum of any and all negative densities computed
            were between :attr:`negativeDensityWarnPercent` (inclusive)
            and :attre:`negativeDensityErrorPercent` (exclusive)

        """
        nr = len(reactionRates)
        nm = len(concentrations.densities)
        nf = len(fissionYields)

        if not nr == nm == nf:
            raise ValueError(
                "Inconsistent number of reaction rates {} to burnable "
                "materials {} and fission yields {}".format(nr, nm, nf)
            )

        zaiOrder = {iso.zai: ix for ix, iso in enumerate(concentrations.isotopes)}

        matrices = starmap(
            self.chain.formMatrix,
            zip(reactionRates, fissionYields, repeat(zaiOrder, nm)),
        )

        inputs = zip(matrices, concentrations.densities, repeat(dtSeconds, nm))

        with multiprocessing.Pool() as p:
            out = p.starmap(self._depsolver, inputs)

        densities = numpy.asarray(out)

        self._checkFixNegativeDensities(densities)

        return CompBundle(concentrations.isotopes, densities)

    def _checkFixNegativeDensities(self, densities):
        """Replace negatives in-place, warning or erroring as appropriate"""
        negativeIndex = densities < 0
        if not negativeIndex.any():
            return

        sumNeg = -densities[negativeIndex].sum()
        negFrac = sumNeg / densities[~negativeIndex].sum()

        if (self._negativeDensityWarn
                < negFrac
                < self._negativeDensityError):
            warnings.warn(
                (f"Replacing negative densities {sumNeg:9.5E} [atoms/b/cm] "
                 f"({negFrac*100:.2f} %)"),
                NegativeDensityWarning,
            )
        elif negFrac >= self._negativeDensityError:
            raise NegativeDensityError(
                f"Sum of negative densities {sumNeg:9.5E} ({negFrac*100:.2f} %) "
                f"exceeded tolerance of {self.negativeDensityErrorPercent*100:.5f} %"
            )

        densities[negativeIndex] = 0.0
