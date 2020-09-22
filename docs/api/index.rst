.. _api-ref:

=========
Reference
=========

.. list-table:: Sub-library breakdown
   :header-rows: 1
   :widths: 20 50


   * - Library
     - Description
   * - :mod:`hydep`
     - Primary user-facing API for model-building, depletion, and 
       orchestrating the coupled simulation
   * - :mod:`hydep.serpent`
     - Interface for the Serpent Monte Carlo code developed and
       distributed by VTT Technical Research Centre of Finland
   * - :mod:`hydep.sfv`
     - Interface for the Spatial Flux Variation (SFV) method for
       prediction change in scalar neutron flux between two state points
   * - :mod:`hydep.lib`
     - Abstract base classes from which much of the project inherits.
       Can be used for developing additional interfaces and data exporters
   * - :mod:`hydep.hdfstore`
     - Interface for writing simulation results to heirarchical data
       format (HDF) files
   * - :mod:`hydep.internal`
     - Developer-focused library containing classes that help with data
       management. End-users will not need to use this library

.. toctree::
    :maxdepth: 1
    :caption: Contents

    main
    serpent
    simplero
    sfv
    hdf
    lib
    internal
    features
    exceptions
