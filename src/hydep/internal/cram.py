"""
CRAM depletion solver

Shadows solvers from OpenMC - MIT Licensed
Copyright: 2011-2019 Massachusetts Institute of Technology and
OpenMC collaborators

https://docs.openmc.org
https://docs.openmc.org/en/stable/pythonapi/deplete.html
https://github.com/openmc-dev/openmc
"""

try:
    from openmc.deplete import Cram16Solver, Cram48Solver
except ImportError:
    from ._cram import Cram16Solver, Cram48Solver
