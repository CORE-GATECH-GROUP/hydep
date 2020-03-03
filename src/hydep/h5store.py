"""
============
HDF5 Storage
============

Currently the primary output format.

Format
======

.. note::

    The current version of this file is ``0.0``

----------
Attributes
----------

The following attributes are written as root level attributes

* ``coarse steps`` ``int`` - Number of coarse time steps

* ``total steps`` ``int`` - Number of total times steps, including
  substeps

* ``isotopes`` ``int`` - Number of isotopes in the depletion chain

* ``burnable materials`` ``int`` - Number of burnable materials in
  the problem

* ``energy groups`` ``int`` - Number of energy groups for flux and
  cross sections

* ``file version`` ``int`` ``(2, )`` - Major and minor version of the file

* ``hydep version`` ``int`` ``(3, )`` [optional] - Version of
  ``hydep`` package

* ``high fidelity solver`` ``str`` [optional] - Name of the high
  fidelity transport solver

* ``reduced order sovler`` ``str`` [optional] - Name of the reduced
  order transport solver

Datasets
--------

* ``/multiplication factor`` ``double`` ``(N_total, 2)`` - Array
  of multiplication factors and absolute uncertainties such that
  ``mf[0, j]`` is the multiplication factor for time point ``j``
  and ``mf[1, j]`` is the associated uncertainty

* ``/fluxes`` ``double`` ``(N_total, N_bumats, N_groups)`` -
  Array of fluxes [n/s] in each burnable material. Note: fluxes are
  normalized to a constant power but are not scaled by volume

* ``/cpu times`` ``double`` ``(N_total, )`` - Array of cpu time [s]
  taken at each transport step, both high fidelity and reduced order

* ``/compositions`` ``double`` ``(N_total, N_bumats, N_isotopes)`` -
  Array of atom densities [#/b-cm] for each material at each point
  in time. The density of isotope ``i`` at point ``j`` in material
  ``m`` is ``c[j, m, i]``.

------
Groups
------

``/fission matrix`` group
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

``/fission matrix/<i>`` group
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

* ``/time/high fidelity`` ``int`` ``(N_total, )`` - Boolean vector
  describing if a specific point corresponds to a high fidelity
  simulation (1) or a reduced order simulation (0)

``/isotopes`` group
-------------------

Group describing isotopes present in the depletion chain, and
their indices in other data sets.

* ``/isotopes/zais`` ``int`` ``(N_isotopes, )`` - Vector
  of isotope ZAI numbers, ordered consistent with the depletion
  chain. More information on isotope ``zais[i]`` can be found
  in the ``/isotopes/i/`` group

``/isotopes/<i>`` group
~~~~~~~~~~~~~~~~~~~~~~~

Contains information for the ``i``-th isotope. Has the following
attributes:

* ``ZAI`` ``int`` - Isotope ``ZZAAAI`` identifier where ``ZZ`` is
 the number of protons, ``AAA`` is the number of protons and neutrons,
 and ``I`` is the metastable state.
* ``name`` ``str`` - Name of this isotope following the Generalized
  Nuclear Data (GND) format.


"""
import typing
import pathlib
import warnings

import numpy
import h5py

from .store import BaseStore


class HdfStore(BaseStore):
    """Write transport and depletion result to HDF files

    Parameters
    ----------
    filename : str, optional
        Name of the file to be written. Default: ``"hydep-results.h5"``
    libver : {"earliest", "latest"}, optional
        Which version of HDF file to write. Passing
        ``"earliest"`` helps with back compatability **with the
        hdf5 library**, while ``"latest"`` may come with
        performance improvements. Default: ``"latest"``
    existOkay : bool, optional
        If ``filename`` exists and is a file, this controls if
        a warning is raised (True) or error (False). The file
        will be overwritten if it exists and ``eixstOkay`` evaluates
        to True.

    Attributes
    ----------
    VERSION : Tuple[int, int]
        Major and minor version of the stored data. Changes to major
        version will reflect new layouts and/or data has been removed.
        Changes to the minor version reflect new data has been added,
        or performance improvements. Scripts that work for ``x.y`` can
        be expected to also work for ``x.z``, but compatability between
        ``a.b`` and ``c.d`` is not guaranteed.

    Raises
    ------
    OSError
        If ``filename`` exists and is not a file
    FileExistsError
        If ``filename`` exists, is a file, and ``existOkay``
        evaluates to ``False``.

    """

    _VERSION = (0, 0)
    _fluxKey = "fluxes"
    _kKey = "multiplication factor"
    _isotopeKey = "isotopes"
    _compKey = "compositions"
    _cputimeKey = "cpu times"
    _timeKey = "time"
    _matKey = "materials"
    _fmtxKey = "fission matrix"

    def __init__(
        self,
        filename: typing.Optional[str] = None,
        libver: typing.Optional[str] = None,
        existOkay: typing.Optional[bool] = True,
    ):

        if libver is None:
            libver = "latest"

        if filename is None:
            filename = "hydep-results.h5"

        fp = pathlib.Path(filename)
        if fp.exists():
            if not fp.is_file():
                raise OSError(f"Result file {fp} exists but is not a file")
            if not existOkay:
                raise FileExistsError(
                    f"Refusing to overwrite result file {fp} since existOkay is True"
                )
            warnings.warn(f"Result file {fp} exists and will be overwritten")

        with h5py.File(fp, mode="w", libver=libver) as h5f:
            h5f.attrs["file version"] = self.VERSION
        self._fp = fp

    @property
    def VERSION(self):
        return self._VERSION

    def beforeMain(self, nhf, ntransport, ngroups, isotopes, burnableIndexes):
        """Called before main simulation sequence

        Parameters
        ----------
        isotopes : tuple of hydep.internal.Isotope
            Isotopes used in the depletion chain
        burnableIndexes : iterable of [int, str]
            Burnable material ids and names ordered how they
            are used across the sequence

        """
        with h5py.File(self._fp, "a") as h5f:
            for src, dest in (
                (nhf, "coarse steps"),
                (ntransport, "total steps"),
                (len(isotopes), "isotopes"),
                (len(burnableIndexes), "burnable materials"),
                (ngroups, "energy groups"),
            ):
                h5f.attrs[dest] = src

            tgroup = h5f.create_group(self._timeKey)
            tgroup.create_dataset("time", (ntransport,))
            tgroup.create_dataset("high fidelity", (ntransport,), dtype="i8")

            h5f.create_dataset(self._kKey, (ntransport, 2))

            h5f.create_dataset(self._cputimeKey, (ntransport,))

            h5f.create_dataset(
                self._fluxKey, (ntransport, len(burnableIndexes), ngroups)
            )

            h5f.create_dataset(
                self._compKey,
                (ntransport, len(burnableIndexes), len(isotopes)),
            )

            isogroup = h5f.create_group(self._isotopeKey)
            zai = numpy.empty(len(isotopes), dtype=int)

            for ix, iso in enumerate(isotopes):
                zai[ix] = iso.zai
                group = isogroup.create_group(str(ix))
                group.attrs["ZAI"] = iso.zai
                group.attrs["name"] = iso.name

            isogroup["zais"] = zai

            materialgroup = h5f.create_group(self._matKey)

            for ix, (matid, name) in enumerate(burnableIndexes):
                group = materialgroup.create_group(str(ix))
                group.attrs["id"] = matid
                group.attrs["name"] = name

    def postTransport(self, timeStep, transportResult) -> None:
        """Store transport results

        Transport results will come both after high fidelity
        and reduced order solutions.

        Parameters
        ----------
        timeStep : hydep.internal.TimeStep
            Point in calendar time from where these results were
            generated
        transportResult : hydep.internal.TransportResult
            Collection of data. Guaranteed to have at least
            a ``flux`` and ``keff`` attribute that are not
            ``None``

        """
        with h5py.File(self._fp, mode="a") as h5f:
            timeindex = timeStep.total
            tgroup = h5f[self._timeKey]
            tgroup["time"][timeindex] = timeStep.currentTime
            tgroup["high fidelity"][timeindex] = 0 if timeStep.substep else 1

            h5f[self._kKey][timeindex] = transportResult.keff

            h5f[self._fluxKey][timeindex] = transportResult.flux

            cputime = transportResult.runTime
            if cputime is None:
                cputime = numpy.nan
            h5f[self._cputimeKey][timeindex] = cputime

            fmtx = transportResult.fmtx
            if fmtx is not None:
                fGroup = h5f.get(self._fmtxKey)
                if fGroup is None:
                    fGroup = h5f.create_group(self._fmtxKey)
                    fGroup.attrs["structure"] = "csr"
                    fGroup.attrs["shape"] = fmtx.shape
                thisG = fGroup.create_group(str(timeindex))
                thisG.attrs["nnz"] = fmtx.nnz
                for attr in {"data", "indices", "indptr"}:
                    thisG[attr] = getattr(fmtx, attr)

    def writeCompositions(self, timeStep, compBundle) -> None:
        """Write (potentially) new compositions

        Parameters
        ----------
        timeStep : hydep.internal.TimeStep
            Point in calendar time that corresponds to the
            compositions, e.g. compositions are from this point
            in time
        compBundle : hydep.internal.CompBundle
            New compositions. Will contain ordering of isotopes and
            compositions ordered consistent with the remainder
            of the sequence and corresponding argument to
            :meth:`beforeMain`

        """
        with h5py.File(self._fp, mode="a") as h5f:
            h5f[self._compKey][timeStep.total] = compBundle.densities
