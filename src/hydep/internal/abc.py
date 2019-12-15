"""
Abstract base classes for extending some internals
"""

try:
    from openmc.deplete.abc import DepSystemSolver
except ImportError:
    from ._abc import DepSystemSolver
