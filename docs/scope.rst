.. _scope:

Scope
=====

:mod:`hydep` does not seek to be a general purpose reactor physics modeling
utility. There are much better programs for that. Instead, a restricted
set of problems is supported at the start. This document will be focused
on presenting the restrictions and assumptions underwhich much of the
framework operates. In the future, some of these may be lifted.

.. _scope-geometry:

Geometry modeling
-----------------

Rather than employ a constructive solid geometry (CSG) representation that
is used in most if not all MC codes. The motivation for this is primarily
expediency, but also for practicallity. While CSG is excellent for ray tracing
and other facets of MC transport, it can be overly complicated for modeling
fuel assemblies in a nodal diffusion code. 

Instead, :mod:`hydep` employs a design model where objects in the API closely
resemble physical structures one might see in a fission reactor code. Namely
fuel pins, axial stacks with some or no variation in properties, and bundles
of assemblies (e.g. super cells). The goal is that a single class,
:class:`hydep.Pin` for example, contains sufficient information that any
transport code can create a corresponding representation using its native
tools. The following subsections will further expand specific restrictions
placed on the geometric modeling inside :mod:`hydep`.

.. seealso::

    :ref:`api-geometry` - API reference for geometric modeling

LWR-style geometries
~~~~~~~~~~~~~~~~~~~~

The decision was made to limit geometries to those typically found in LWRs.
Specifically, the following restrictions are in place:

1. Fuel pins are constructed of concentric cylinders, each ring containing a
   single homogenized material
2. The transverse direction of the fuel pin aligns with the z-axis
3. Fuel pins can be arranged in structured rectangular lattices, tiled across
   the x-y plane 
4. Vertical stacks of fuel pins and/or rectangular lattices can be created to
   create 3D models with either constant or varying axial properties (e.g. various
   enrichment zones, coolant density profiles, etc.)
5. Larger super-cells can be created further by tiling vertical stacks (of fuel
   pins and/or lattices)
6. Lattice and stack elements can be filled with a single material instead of
   any of the aforementioned structures
7. The final geometry must be either a 2-D rectangle or 3-D rectangular prism,
   with primary axes aligning with the x, y, and z coordinate system

With these rules in place, one can create models from a single 2-D or 3-D pincell,
to the active region of a PWR with or without axial reflectors.

Minimal support for extra-core structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By following the rules from the previous section, there are some clear limitations.
Common structures like axial grids, structural baffles, core barrels, and pressure
vessels are not easily supported. While one could create a hyper-fine Cartesian
grid to approximate the curvature of a pressure vessel, this is not advised.

Single energy group for internal data handling
----------------------------------------------

While the underlying transport codes can use any or no energetic
approximations, the resulting scalar flux, reaction rates,
micro- and macroscopic cross sections **must** be provided back
to the framework in a single energy group. This is motivated by the
fact that reaction rates in depletion require a single energy group.
Some reduced-order codes will surely perform better for specific problems
if provided multi-group cross sections (e.g. diffusion codes), but this
is not under consideration at this time.

