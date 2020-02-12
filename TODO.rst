Big open items
==============

* Depletion framework
* SFV solver
* Storing of results in a nice framework
* Documentation outside of docstrings
* Non-constant fission yields

TODOs
=====

* Status reporting 
* Standardize microXS vs microXs
* Guard against hooks that aren't supported by solvers
* Validate configuration options to avoid typos
* Resolve how hydep.Pin objects are handled by numpy
* Remove most assert statements in favor of actual checks
* Make depletion solver / CRAM order configurable
* Find a way to add regression test file (setup, post-process) into
  the repository
* Make all tests less flaky. Failures can fill up the Material or
  Universe register. Some of the writer tests require strict orderings
  of Materials and Universes.
* Use procedurally generated reference files, e.g. with Jinja, for
  testing Serpent writer

Serpent
-------

* Writing metastable isotopes
* Writing / working with metastable isotopes
* Get microscopic cross sections without second transport simulation
* Improved parsing of error at output
* Pull and check model.bounds, not model.root.bounds

Caveats
=======

* Serpent fluxes, microscopic xs, and macroscopic xs are one group
* Fetching microscopic reaction xs with Serpent requires depletion

Wish list
=========
* Verbosity control
* Multigroup
* Reflective geometry
* OpenMC compatibility?
* Helper function for making pins with automatic subdivision
  a. la. openmc.model.pin
* Helper for making UO2-like materials
* Weakref for Isotopes
* Share some code between CartesianLattice and LatticeStack
