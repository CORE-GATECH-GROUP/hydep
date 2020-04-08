.. currentmodule:: hydep.hdfstore

.. _api-hdf:

=============
HDF Interface
=============

This is the primary way to store transport and depletion results generated
through the simulation. It requires the
`h5py module <https://docs.h5py.org/>`_ module.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: myclass.rst

    HdfStore
    HdfProcessor

The following :class:`~enum.Enum` classes are provided to provide a more consistent
and programmatic way to index directly in to the HDF files. It is recommended to
use the :class:`HdfProcessor`, as it provides additional convenience methods on top
of acting like an HDF file anyway.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: myclass.rst

    HdfStrings
    HdfSubStrings
    HdfAttrs

.. _hdf-format:

Format
======

.. note::

    The current version of this file is ``0.1``

----------
Attributes
----------

The following attributes are written as root level attributes

* ``coarseSteps`` ``int`` - Number of coarse time steps

* ``totalSteps`` ``int`` - Number of total times steps, including
  substeps. Denoted as ``N_total`` through this document.

* ``isotopes`` ``int`` - Number of isotopes in the depletion chain

* ``burnableMaterials`` ``int`` - Number of burnable materials in
  the problem. Denoted as ``N_bumats`` through this document.

* ``energyGroups`` ``int`` - Number of energy groups for flux and
  cross sections. Denoted as ``N_groups`` through this document.

* ``fileVersion`` ``int`` ``(2, )`` - Major and minor version of the file

* ``hydepVersion`` ``int`` ``(3, )`` - Version of ``hydep`` package

Datasets
--------

* ``/multiplicationFactor`` ``double`` ``(N_total, 2)`` - Array
  of multiplication factors and absolute uncertainties such that
  ``mf[j, 0]`` is the multiplication factor for time point ``j``
  and ``mf[j, 1]`` is the associated uncertainty

* ``/fluxes`` ``double`` ``(N_total, N_bumats, N_groups)`` -
  Array of fluxes [n/cm3/s] in each burnable material. Note: fluxes are
  normalized to the power for the given depletion step

* ``/cpuTimes`` ``double`` ``(N_total, )`` - Array of cpu time [s]
  taken at each transport step, both high fidelity and reduced order

* ``/compositions`` ``double`` ``(N_total, N_bumats, N_isotopes)`` -
  Array of atom densities [#/b-cm] for each material at each point
  in time. The density of isotope ``i`` at point ``j`` in material
  ``m`` is ``c[j, m, i]``.

------
Groups
------

``/fissionMatrix`` group
-------------------------

If the fission matrix is present on at least one transport result,
then this group will be created. The following attributes will be
written to this group:

* ``structure`` ``str`` - Sparsity structure of the fission matrices.
  Currently ``"csr"``, indicating a Compressed Sparse Row storage.

All matrices will have shape ``(N_bumats, N_bumats)``, but their
structure may change from step to step.

The fission matrix generated at step ``i`` will be written into a
subgroup.

``/fissionMatrix/<i>`` group
-----------------------------

The subgroup will have the following attributes:

* ``nnz`` ``int`` - Number of non-zero elements in this matrix.

and datasets:

* ``indptr`` ``int`` ``(N_bumats + 1, )`` - Pointer vector
  indicating where non-zero elements for row ``r`` are stored
  in ``data``

* ``indices`` ``int``  (``nnz, )`` - Columns with non-zero
  data such that row ``r`` has non-zero entries in columns
  ``indices[indptr[r]:indptr[r+1]]``

* ``data`` ``double`` ``(nnz, )`` - Vector of non-zero data
  such that non-zero values in row ``r`` are
  ``data[indptr[r]:indptr[r + 1]]``, located corresponding to
  ``indices`` vector.

``/time`` group
---------------

* ``/time/time`` ``double`` ``(N_total, )`` - Vector of points in
  calendar time [s]

* ``/time/highFidelity`` ``bool`` ``(N_total, )`` - Boolean vector
  describing if a specific point corresponds to a high fidelity
  simulation (True) or a reduced order simulation (False)

``/isotopes`` group
-------------------

Group describing isotopes present in the depletion chain, and
their indices in other data sets.

* ``/isotopes/zais`` ``int`` ``(N_isotopes, )`` - Vector
  of isotope ZAI numbers, ordered consistent with the depletion
  chain.
* ``/isotopes/names`` ``S`` ``(N_isotopes,)`` - Vector with isotope
   names, ordered consistent with the depletion chain

``/materials`` group
--------------------

Group describing ordering of burnable materials and their names.
Written to be a consistent ordering across fluxes and compositions

* ``/materials/ids`` ``int`` ``(N_bumats, )`` - Vector with burnable
  material ID
* ``/materials/names`` ``S`` ``(N_bumats, )`` - Vector with material
  names
