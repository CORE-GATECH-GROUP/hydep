.. _api-internal:

================
Internal Helpers
================

Feature Control
===============

.. currentmodule:: hydep.internal.features

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: myclass.rst

    Feature
    FeatureCollection

.. currentmodule:: hydep.internal

Isotopics
=========

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: myclass.rst

    ZaiTuple
    Isotope
    ReactionTuple
    DecayTuple

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: myfunction.rst

    getIsotope
    allIsotopes


Other Helpers
=============

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: myclass.rst

    TimeStep
    TransportResult
    XsIndex
    TemporalMicroXs
    MicroXsVector
    XsTimeMachine
    Boundaries
    CompBundle


``openmc``-inspired
===================

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: myclass.rst

    Cram16Solver
    Cram48Solver
    FissionYieldDistribution
    FissionYield

Time Traveler Bases
===================

.. currentmodule:: hydep.internal.timetravel

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: myclass.rst

    TimeTraveler
    CachedTimeTraveler
