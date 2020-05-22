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
