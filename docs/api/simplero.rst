.. currentmodule:: hydep.simplero

.. module:: hydep.simplerom

.. _api-simplero:

Simple reduced-order solver
===========================

This module defines what is basically a no-operation reduced
order transport solver. It simply returns the flux provided
by the most recent high-fidelity transport solution. One
could use this class to create a substepping procedure like
that described in [Isotalo2011b]_. It is also used to generate
solutions that don't depend on any substepping.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: myclass.rst

    SimpleROSolver

References
----------

.. [Isotalo2011b] `Isotalo and Aarnio, Annals. Nuc. Ene. (2011) <https://doi.org/10.1016/j.anucene.2011.07.012>`_