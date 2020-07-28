.. _api-serpent:

.. currentmodule:: hydep.serpent

.. module:: hydep.serpent

=================
Serpent Interface
=================

Primarily, users are only expected to interact with
:class:`SerpentSolver` and :class:`CoupledSerpentSolver`. The latter
requires a modified version of Serpent 2.1.31 that has yet to be
distributed to the user community.

Solvers
=======

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: myclass.rst

    SerpentSolver
    CoupledSerpentSolver

Support
=======

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: myclass.rst

    SerpentWriter
    SerpentRunner
    SerpentProcessor
    ExtDepWriter
    ExtDepRunner


Fission product yields
----------------------

The following classes define and provide interfaces for obtaining
fission product yields. These are controlled by
:attr:`hydep.SerpentSettings.fpyMode`.

.. currentmodule:: hydep.serpent.processor


.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: myclass.rst

    ConstantFPYHelper
    WeightedFPYHelper
    FPYHelper