"""
Depletion manager

Control time-steps, material divisions, depletion, etc
"""

import numbers
from collections import defaultdict
from collections.abc import Sequence, Iterable
from itertools import repeat, starmap
import multiprocessing

import numpy

from hydep import BurnableMaterial, DepletionChain
from hydep.constants import SECONDS_PER_DAY
from hydep.typed import TypedAttr, IterableOf
from hydep.internal import TemporalMicroXs, MicroXsVector, Cram16Solver, CompBundle
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
                    "Non-integer preliminary steps not allowed: {}".format(
                        type(numPreliminary)
                    )
                )
            elif not (0 <= numPreliminary < len(self.timesteps)):
                raise ValueError(
                    "Number of preliminary steps must be between [0, {}), "
                    "not {}".format(len(self.timesteps), numPreliminary)
                )
            self._nprelim = numPreliminary
        self._microXS = None

        # TODO Make CRAM solver configurable property
        # NOTE: Must be an importable function that we can dispatch
        # through multiprocessing
        self._depsolver = Cram16Solver

    def _validatePowers(self, power):
        if isinstance(power, numbers.Real):
            if power <= 0:
                raise ValueError("Power must be positive, not {}".format(power))
            return repeat(power, len(self.timesteps))
        elif isinstance(power, Sequence):
            if len(power) != len(self.timesteps):
                raise ValueError(
                    "Number of powers {} differ from steps {}".format(
                        len(power), len(self.timesteps)
                    )
                )
            for p in power:
                if not isinstance(p, numbers.Real) or p <= 0:
                    raise TypeError(
                        "Power must be positive real, or vector of positive real. "
                        "Found {}".format(p)
                    )
            return power
        else:
            raise TypeError(
                "Power must be positive real, or vector of positive real, "
                "not {}".format(type(power))
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

    # TODO don't use numberKeep, just store all of them.
    # Pass control to whatever is calling this
    def setMicroXS(self, mxs, times, numberKeep=None, polyorder=3):
        """Construct the initial microscopic cross section storage

        Parameters
        ----------
        mxs : Iterable[hydep.internal.MicroXsVector]
            Initial set of cross sections to store
        times : Iterable[float]
            Points in calendar time at which the cross sections
            were generated
        numberKeep : Optional[int]
            Store only N values
        polyOrder : Optional[int]
            Fitting order

        """
        if not isinstance(mxs, Iterable):
            raise TypeError(
                "mxs must be Iterable of {}, not {}".format(
                    MicroXsVector.__name__, type(mxs)
                )
            )

        if not isinstance(times, Iterable):
            raise TypeError(
                "times must be Iterable of real, not {}".format(type(times))
            )

        if len(times) != len(mxs):
            raise ValueError(
                "Number of time points {} not equal to number of cross "
                "sections {}".format(len(times), len(mxs))
            )

        assert isinstance(polyorder, numbers.Integral) and polyorder >= 0

        if numberKeep is not None:
            assert polyorder < numberKeep
            if not isinstance(numberKeep, numbers.Integral):
                raise TypeError("Number of steps to keep must be integer")
            elif 0 > numberKeep:
                raise ValueError("Number of steps to keep must be positive")
            mxs = mxs[-numberKeep:]
            times = times[-numberKeep:]

        self._microXS = self._makeMicroXS(mxs, times, numberKeep, polyorder)

    @staticmethod
    def _makeMicroXS(mxs, times, maxSteps, polyorder):
        out = []
        for microvector in mxs[0]:
            out.append(
                TemporalMicroXs.fromMicroXsVector(
                    microvector, times[0], maxlen=maxSteps, order=polyorder
                )
            )

        # Assume all incoming microxs have the same configurations
        # (zai, rxn, zptr orderings) for each time step
        # TODO Guard against ^^^
        materialXs = defaultdict(list)
        for timexs in mxs[1:]:
            for matix, matxs in enumerate(timexs):
                materialXs[matix].append(matxs)

        for matix, matvector in enumerate(out):
            matvector.extend(times[1:], materialXs[matix])

        return tuple(out)

    def getReactionRatesAt(self, time, fluxes):
        """Compute one-group reaction rates in burnable regions

        Parameters
        ----------
        time : float
            Time at which the reaction rates are expected
        fluxes : numpy.ndarray
            Flux in each burnable region such that ``fluxes[i, g]``
            is the ``g``-th group flux in region ``i``.

        Returns
        -------
        Tuple[MicroXsVector...]
            Stand-in class for reaction rates in each burnable region

        Raises
        ------
        NotImplementedError
            If fluxes are not presented with a single energy group
        """
        rxnXS = [mxs(time) for mxs in self._microXS]
        if fluxes.shape[1] == 1:
            # Special treatement for 1-group data
            rates = [f * micro.mxs[:, 0] for f, micro in zip(fluxes.flat, rxnXS)]
        else:
            raise NotImplementedError("Multigroup collapsing not implemented yet")

        volumes = (m.volume for m in self.burnable)

        return tuple(
            MicroXsVector(m.zai, m.zptr, m.rxns, r / v)
            for m, r, v in zip(rxnXS, rates, volumes)
        )

    def deplete(self, start, dtSeconds, txResult):
        """Deplete all burnable materials

        Parameters
        ----------
        start : float
            Starting point in time. Reaction rates will be obtained
            at this point using :meth:`getReactionRatesAt` and
            compositions will be depleted with compositions at
            this point in time
        dtSeconds : float
            Length of depletion interval in seconds
        txResult : hydep.internal.TransportResult
            Previous transport result containing fluxes and fission yields

        Returns
        -------
        list of numpy.ndarray
            Compositions for each region such that ``out[i][j]`` is the
            final concentration of isotope :attr:`hydep.DepletionChain.zaiOrder`
            position ``j`` for burnable region ``i``

        """
        assert self.burnable is not None
        assert self._microXS is not None

        reactionRates = self.getReactionRatesAt(start, txResult.flux)
        assert len(reactionRates) == len(self.burnable) == len(txResult.fissionYields)
        concentrations = (m.asVector(order=self.chain.zaiOrder) for m in self.burnable)

        matrices = starmap(self.chain.formMatrix,
                           zip(reactionRates, txResult.fissionYields))

        inputs = zip(matrices, concentrations, repeat(dtSeconds, len(self.burnable)))

        with multiprocessing.Pool() as p:
            out = p.starmap(self._depsolver, inputs)

        return CompBundle(tuple(self.chain), out)
