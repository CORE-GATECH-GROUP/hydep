"""
Depletion manager

Control time-steps, material divisions, depletion, etc
"""

import numbers
from collections.abc import Sequence
from itertools import repeat, starmap
import multiprocessing

import numpy

from hydep import BurnableMaterial, DepletionChain
from hydep.constants import SECONDS_PER_DAY
from hydep.typed import TypedAttr, IterableOf
from hydep.internal import Cram16Solver, CompBundle
from hydep.internal.features import FeatureCollection, MICRO_REACTION_XS, FISSION_YIELDS


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
    numPreliminary : int, optional
        Number of coarse depletion steps to take before engaging in
        coupled behavior. Useful for approaching some equilibrium value
        with smaller steps with the high fidelity code

    Attributes
    ----------
    chain : hydep.DepletionChain
        Depletion chain
    timesteps : numpy.ndarray
        Length of time [s] for each coarse depletion step
    power : numpy.ndarray
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
    """

    _nExtrapSteps = 3  # TODO Make this configurable

    # TODO Make depletion chain configurable property
    chain = TypedAttr("chain", DepletionChain)
    _burnable = IterableOf("burnable", BurnableMaterial, allowNone=True)

    def __init__(self, chain, daysteps, power, numPreliminary=0):
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

        # TODO Make CRAM solver configurable property
        # NOTE: Must be an importable function that we can dispatch
        # through multiprocessing
        self._depsolver = Cram16Solver

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
                if not isinstance(p, numbers.Real) or p <= 0:
                    raise TypeError(
                        "Power must be positive real, or vector of positive real. "
                        f"Found {p}"
                    )
            return power
        else:
            raise TypeError(
                "Power must be positive real, or vector of positive real, "
                f"not {type(power)}"
            )

    @property
    def burnable(self):
        return self._burnable

    @property
    def needs(self):
        return FeatureCollection({MICRO_REACTION_XS, FISSION_YIELDS})

    @property
    def numPreliminary(self):
        return self._nprelim

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

    def beforeMain(self, model):
        """Check that all materials have volumes and set indexes

        Parameters
        ----------
        model : hydep.Model
            Problem to be solved. Must contain at least one
            :class:`hydep.BurnableMaterial`

        """
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

    def deplete(self, dtSeconds, reactionRates, fissionYields):
        """Deplete all burnable materials

        Parameters
        ----------
        dtSeconds : float
            Length of depletion interval in seconds
        reactionRates : iterable of hydep.internal.MicroXsVector
            Stand in for microscopic reaction rates in each burnable
            material
        fissionYields : iterable of hydep.internal.FissionYield
            Fission yields in each burnable material

        Returns
        -------
        hydep.internal.CompBundle
            New compositions for each burnable material and the isotope
            ordering

        """
        if not len(reactionRates) == len(self.burnable) == len(fissionYields):
            raise ValueError(
                "Inconsistent number of reaction rates {} to burnable "
                "materials {} and fission yields {}".format(
                    *(len(x) for x in (reactionRates, self.burnable, fissionYields))
                )
            )
        concentrations = (m.asVector(order=self.chain.zaiOrder) for m in self.burnable)

        matrices = starmap(
            self.chain.formMatrix, zip(reactionRates, fissionYields)
        )

        inputs = zip(matrices, concentrations, repeat(dtSeconds, len(self.burnable)))

        with multiprocessing.Pool() as p:
            out = p.starmap(self._depsolver, inputs)

        return CompBundle(tuple(self.chain), out)
