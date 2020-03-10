"""
Serpent high fidelity transport solver interface

Primarily beneficial classes:

.. autosummary::
    :toctree: generated
    :nosignatures:

    SerpentSolver
    CoupledSerpentSolver

"""
try:
    import serpentTools
except ImportError:
    raise ImportError(
        "serpentTools required for Serpent interface. Install as extra "
        "with pip install <options> hydep[serpent]")

from .writer import SerpentWriter, ExtDepWriter
from .runner import SerpentRunner, ExtDepRunner
from .processor import SerpentProcessor
from .solver import SerpentSolver, CoupledSerpentSolver
