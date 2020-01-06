"""
Depletion manager

Control time-steps, material divisions, depletion, etc
"""

import numbers
from collections.abc import Sequence
from itertools import repeat

import numpy

from hydep import BurnableMaterial, DepletionChain
from hydep.typed import TypedAttr, IterableOf
from hydep.internal.features import FeatureCollection, MICRO_REACTION_XS


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

    chain = TypedAttr("chain", DepletionChain)
    _burnable = IterableOf("burnable", BurnableMaterial, allowNone=True)

    def __init__(self, chain, daysteps, power, numPreliminary=0):
        self.chain = chain

        daysteps = numpy.asarray(daysteps, dtype=float)
        if len(daysteps.shape) > 1:
            raise TypeError("Day steps must be vector, not array")
        self.timesteps = tuple(daysteps * 86400)

        self.powers = tuple(self._validatePowers(power))

        self._burnable = None

        if numPreliminary is None:
            self._nprelim = 0
        else:
            if not isinstance(numPreliminary, numbers.Integral):
                raise TypeError(
                    "Non-integer preliminary steps not allowed: {}".format(
                        type(numPreliminary)))
            elif not (0 <= numPreliminary < len(self.timesteps)):
                raise ValueError(
                    "Number of preliminary steps must be between [0, {}), "
                    "not {}".format(len(self.timesteps), numPreliminary))
            self._nprelim = numPreliminary

    def _validatePowers(self, power):
        if isinstance(power, numbers.Real):
            if power <= 0:
                raise ValueError("Power must be positive, not {}".format(power))
            return repeat(power, len(self.timesteps))
        elif isinstance(power, Sequence):
            if len(power) != len(self.timesteps):
                raise ValueError(
                    "Number of powers {} differ from steps {}".format(
                        len(power), len(self.timesteps)))
            for p in power:
                if not isinstance(p, numbers.Real) or p <= 0:
                    raise TypeError(
                        "Power must be positive real, or vector of positive real. "
                        "Found {}".format(p))
            return power
        else:
            raise TypeError(
                "Power must be positive real, or vector of positive real, "
                "not {}".format(type(power)))

    @property
    def burnable(self):
        return self._burnable

    @property
    def needs(self):
        return FeatureCollection({MICRO_REACTION_XS})

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
        return zip(self.timesteps[:self._nprelim], self.powers[:self._nprelim])

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
        return zip(self.timesteps[self._nprelim:], self.powers[self._nprelim:])

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
            raise ValueError("No burnable materials found in {}".format(model))

        for ix, mat in enumerate(burnable):
            if mat.volume is None:
                raise AttributeError(
                    "{} {} does not have a volume set".format(mat.__class__, mat)
                )
            mat.index = ix

        self._burnable = burnable

    def checkCompatibility(self, hf):
        # Check for compatibility with high fidelity solver
        pass

    def pushResults(self, time, results):
        pass

    def deplete(self, time):
        pass

    def beforeROM(self, time):
        pass
