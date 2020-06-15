import typing

from .lib import Integrator
from .internal import MaterialDataArray


class PredictorIntegrator(Integrator):
    """Employs the standard predictor time-integration scheme

    EOS compositions are computed by depleting across
    the time step with a single step and no additional reduced
    order solutions.

    Parameters
    ----------
    model : hydep.Model
        Representation of the problem geometry
    hf : hydep.lib.HighFidelitySolver
        High fidelity solver to be executed at the beginning
        of each coarse step, and the final EOL point
    ro : hydep.lib.ReducedOrderSolver
        Reduced order solver to be executed at the substeps,
        and potentially any intermediate time points
    dep : hydep.Manager
        Depletion manager, including access to depletion chain
        and depletion solver
    store : hydep.lib.BaseStore, optional
        Instance responsible for writing transport and depletion
        result data. If not provided, will be set to
        :class:`hydep.hdfstore.HdfStore`

    Attributes
    ----------
    model : hydep.Model
        Representation of the problem geometry
    hf : hydep.lib.HighFidelitySolver
        High fidelity solver to be executed at the beginning
        of each coarse step, and the final EOL point
    ro : hydep.lib.ReducedOrderSolver
        Reduced order solver to be executed at the substeps,
        and potentially any intermediate time points
    dep : hydep.Manager
        Depletion manager, including access to depletion chain
        and depletion solver
    store : hydep.lib.BaseStore or None
        Instance responsible for writing transport and depletion
        result data. If not provided, will be set to
        :class:`hydep.hdfstore.HdfStore`
    settings : hydep.Settings
        Simulation settings. Can be updated directly, or
        through :meth:`configure`

    """
    def __call__(
        self,
        timestep: "hydep.internal.TimeStep",
        dt: float,
        compositions: "hydep.internal.CompBundle",
        flux: "numpy.ndarray",
        fissionYields: typing.List[
            typing.Dict[int, "hydep.internal.FissionYield"]
        ],
    ) -> "hydep.internal.CompBundle":
        return self.dep.deplete(
            dt,
            compositions,
            self._xs.getReactionRatesAt(timestep.currentTime, flux),
            fissionYields,
        )


class CELIIntegrator(Integrator):
    def __call__(
        self,
        timestep: "hydep.internal.TimeStep",
        dt: float,
        compositions: "hydep.internal.CompBundle",
        flux: "numpy.ndarray",
        fissionYields: typing.List[
            typing.Dict[int, "hydep.internal.FissionYield"]
        ],
    ) -> "hydep.internal.CompBundle":

        # Predictor
        bosRR = self._xs.getReactionRatesAt(timestep.currentTime, flux)
        eosComp = self.dep.deplete(dt, compositions, bosRR, fissionYields)

        eosFlux, _time = self.ro.intermediateSolve(
            timestep, eosComp, self._xs.at(timestep.currentTime + dt)
        )
        eosRR = self._xs.getReactionRatesAt(timestep.currentTime + dt, eosFlux)

        # Get average reaction rates for corrector step
        avgRR = MaterialDataArray.fromLinearCombination(
            (0.5, bosRR), (0.5, eosRR)
        )

        return self.dep.deplete(
            dt,
            compositions,
            avgRR,
            fissionYields,
        )


class RK4Integrator(Integrator):
    def __call__(
        self,
        timestep: "hydep.internal.TimeStep",
        dt: float,
        comp0: "hydep.internal.CompBundle",
        flux0: "numpy.ndarray",
        fissionYields: typing.List[
            typing.Dict[int, "hydep.internal.FissionYield"]
        ],
    ) -> "hydep.internal.CompBundle":

        midpoint = timestep.currentTime + 0.5*dt

        # Deplete out to mid point
        rr0 = self._xs.getReactionRatesAt(timestep.currentTime, flux0)
        comp1 = self.dep.deplete(0.5 * dt, comp0, rr0, fissionYields)

        flux1, t1 = self.ro.intermediateSolve(
            timestep, comp1, self._xs.at(midpoint)
        )
        # Deplete to midpoint using predicted midpoint reaction rates
        rr1 = self._xs.getReactionRatesAt(midpoint, flux1)
        comp2 = self.dep.deplete(0.5*dt, comp0, rr1, fissionYields)

        flux2, t2 = self.ro.intermediateSolve(
            timestep, comp2, self._xs.at(midpoint)
        )

        # Deplete to EOS with corrected midpoint reaction rates
        rr2 = self._xs.getReactionRatesAt(midpoint, flux2)
        comp3 = self.dep.deplete(dt, comp0, rr2, fissionYields)

        flux3, t3 = self.ro.intermediateSolve(
            timestep, comp3, self._xs.at(timestep.currentTime + dt)
        )

        rr3 = self._xs.getReactionRatesAt(timestep.currentTime + dt, flux3)

        # Get average reaction rates to deplete across entire interval

        avgrr = MaterialDataArray.fromLinearCombination(
            (1, rr0), (2, rr1), (2, rr2), (1, rr3)
        )
        avgrr /= 6

        return self.dep.deplete(dt, comp0, avgrr, fissionYields)
