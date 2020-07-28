.. _intro:

Introduction
============

:mod:`hydep` seeks to provide a framework for coupling one or two neutron
transport codes with a depletion solver, regardless of the depletion
capabilities and fidelity of either code. This is done to enable a 
first-of-its-kind hybrid transport-depletion sequence using mixed-fidelity
transport solutions. For instance, a continuous energy Monte Carlo code
and a method of characteristics code could, in theory, be linked together
using this package, with a custom depletion solver used to update compositions
as the simulation progresses.

.. note::

    As of the time of this writing, interfaces for the Serpent
    Monte Carlo code [Serpent]_, a no-op reduced order solver, and the
    spatial variation method [SFV]_


Motivation
----------

1. Monte Carlo (MC) particle transport codes are considered the gold
   standard when modeling nuclear systems, e.g. fission or fusion reactors
2. When solving a single statepoint, MC codes can be computationally burdensome
3. In order to model a lengthy operational period, e.g. fuel cycle on the order
   of 1-2 years, one must provide transport solutions to a depletion code in
   order to model the production, destruction, and transmutation of isotopes
4. Due to numeric and physical difficulties associated with the non-linear
   transport+depletion coupling, depletion steps are usually quite small for
   simple schemes (order of days to few weeks)
5. Employing advanced time-integration schemes can extend the step size without
   sacrificing sufficient accuracy nor stability

Can we obtain the accuracy and stability of small depletion steps without
incurring a large computational burden by trading some transport solutions
with a faster reduced-order solution?


References
----------

.. [Serpent] `Leppaenen et. al, Annals. Nuc. Ene. (2015) <https://doi.org/10.1016/j.anucene.2014.08.024>`_

.. [SFV] `Johnson and Kotlyar, Nuc. Sci. Eng. (2019) <https://doi.org/10.1080/00295639.2019.1661171>`_