"""
Primary class for handling geometry and material information
"""
import warnings

from .integrators import PredictorIntegrator


class Problem(PredictorIntegrator):
    """Main linkage between geometry, depletion, and transport

    .. warning::

        REMOVE THIS FROM ALL YOUR SCRIPTS DREW.
        USE THE PREDICTOR OR BETTER YET SOMETHING
        WITH A HIGHER ORDER

    Parameters
    ----------
    model : hydep.Model
        Full representation of the geometry
    hf : hydep.lib.HighFidelitySolver
        High fidelity transport solver
    rom : hydep.lib.ReducedOrderSolver
        Reduced order solver of choice
    dep : hydep.Manager
        Depletion interface
    store : hydep.lib.BaseStore, optional
        Object responsible for data storage

    Attributes
    ----------
    model : hydep.Model
        Geometry interface
    hf : hydep.lib.HighFidelitySolver
        High fidelity transport solver
    rom : hydep.lib.ReducedOrderSolver
        Reduced order solver of choice
    dep : hydep.Manager
        Depletion interface
    store : hydep.lib.BaseStore or None
        Object responsible for data storage. If ``None`` by
        the time :meth:`problem` is started, then
        :class:`hydep.HdfStore` will be created and assigned
        here
    settings : hydep.Settings
        Settings interface

    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "hydep.Problem is deprecated and should not be used. "
            "Remove this and use hydep.PredictorIntegrator or "
            "a higher order scheme.",
            PendingDeprecationWarning,
        )
        super().__init__(*args, **kwargs)
