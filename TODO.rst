Big open items
==============

* Documentation outside of docstrings

TODOs
=====

* Standardize microXS vs microXs
* Guard against hooks that aren't supported by solvers
* Using vectorized / subprocess-managed routines for SFV macroxs
  reconstruction
* Use vectorized / subprocess-managed routines when projecting
  MicroXsVector on TemporalMicroXs
* Resolve how hydep.Pin objects are handled by numpy
* Remove most assert statements in favor of actual checks
* Find a way to add regression test file (setup, post-process) into
  the repository
* Make all tests less flaky. Failures can fill up the Material or
  Universe register. Some of the writer tests require strict orderings
  of Materials and Universes.

Serpent
-------

* Improved parsing of error at output
* Support for writing outer materials for lattice stack

Caveats
=======

* Serpent fluxes, microscopic xs, and macroscopic xs are one group
* Fetching microscopic reaction xs with Serpent requires depletion
  unless using CoupledSerpentSolver. This class relies on a patched
  version of Serpent that the author intends to distribute to the
  user community.

Wish list
=========

* Multigroup
* Reflective geometry
* OpenMC compatibility?
* Helper function for making pins with automatic subdivision
  a. la. openmc.model.pin
* Helper for making UO2-like materials
* Weakref for Isotopes
* Share some code between CartesianLattice and LatticeStack
* Use procedurally generated reference files, e.g. with Jinja, for
  testing Serpent writer
* Add other simple ReducedOrderSolvers that interpolate / extrapolate
  flux?
* Improve isotope interface. Slapped an lru_cache on the getIsotope function.
  For a 3x3 pin cell w/ Gad, got ~3800 hits, 17 billion (!!) misses
* Isotopes hashable? They are mutable, in the sense their reactions change.
  But they are hashed based on their hopefully unique and immutable ZAI tuple
