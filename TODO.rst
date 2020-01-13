Big open items
==============

* Depletion framework
* SFV solver
* Storing of results in a nice framework
* Documentation outside of docstrings
* Non-constant fission yields

TODOs
=====

* Mark isotopes in materials as requiring S(a, b)
* Status reporting 
* Guard against hooks that aren't supported by solvers
* Validate configuration options to avoid typos
* Resolve how hydep.Pin objects are handled by numpy
* Remove most assert statements in favor of actual checks
* Make depletion solver / CRAM order configurable

Serpent
-------

* Writing metastable isotopes
* Writing / working with metastable isotopes
* Incorporate thermal scattering
* Control over ace, decay, and nfy libraries
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
