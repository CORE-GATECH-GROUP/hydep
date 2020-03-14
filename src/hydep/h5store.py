"""
============
HDF5 Storage
============

Currently the primary output format.

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
  substeps

* ``isotopes`` ``int`` - Number of isotopes in the depletion chain

* ``burnableMaterials`` ``int`` - Number of burnable materials in
  the problem

* ``energyGroups`` ``int`` - Number of energy groups for flux and
  cross sections

* ``fileVersion`` ``int`` ``(2, )`` - Major and minor version of the file

* ``hydepVersion`` ``int`` ``(3, )`` - Version of ``hydep`` package

Datasets
--------

* ``/multiplicationFactor`` ``double`` ``(N_total, 2)`` - Array
  of multiplication factors and absolute uncertainties such that
  ``mf[0, j]`` is the multiplication factor for time point ``j``
  and ``mf[1, j]`` is the associated uncertainty

* ``/fluxes`` ``double`` ``(N_total, N_bumats, N_groups)`` -
  Array of fluxes [n/s] in each burnable material. Note: fluxes are
  normalized to a constant power but are not scaled by volume

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

"""
import numbers
import typing
import pathlib
from collections.abc import Mapping
import bisect

import numpy
import h5py
from scipy.sparse import csr_matrix

import hydep
from hydep.constants import SECONDS_PER_DAY
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
        Raise an error if ``filename`` already exists. Otherwise
        silently overwrite an existing file

    Attributes
    ----------
    VERSION : Tuple[int, int]
        Major and minor version of the stored data. Changes to major
        version will reflect new layouts and/or data has been removed.
        Changes to the minor version reflect new data has been added,
        or performance improvements. Scripts that work for ``x.y`` can
        be expected to also work for ``x.z``, but compatability between
        ``a.b`` and ``c.d`` is not guaranteed.
    fp : pathlib.Path
        Read-only attribute with the absolute path of the intended result
        result file

    Raises
    ------
    OSError
        If ``filename`` exists and is not a file
    FileExistsError
        If ``filename`` exists, is a file, and ``existOkay``
        evaluates to ``False``.

    """

    _VERSION = (0, 1)
    _fluxKey = "fluxes"
    _kKey = "multiplicationFactor"
    _isotopeKey = "isotopes"
    _compKey = "compositions"
    _cputimeKey = "cpuTimes"
    _timeKey = "time"
    _matKey = "materials"
    _fmtxKey = "fissionMatrix"

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

        fp = pathlib.Path(filename).resolve()

        if fp.exists():
            if not fp.is_file():
                raise OSError(f"Result file {fp} exists but is not a file")
            if not existOkay:
                raise FileExistsError(
                    f"Refusing to overwrite result file {fp} since existOkay is True"
                )

        with h5py.File(fp, mode="w", libver=libver) as h5f:
            h5f.attrs["fileVersion"] = self.VERSION
            h5f.attrs["hydepVersion"] = tuple(
                int(x) for x in hydep.__version__.split(".")[:3]
            )
        self._fp = fp

    @property
    def fp(self):
        return self._fp

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
                (nhf, "coarseSteps"),
                (ntransport, "totalSteps"),
                (len(isotopes), "isotopes"),
                (len(burnableIndexes), "burnableMaterials"),
                (ngroups, "energyGroups"),
            ):
                h5f.attrs[dest] = src

            tgroup = h5f.create_group(self._timeKey)
            tgroup.create_dataset("time", (ntransport,))
            tgroup.create_dataset("highFidelity", (ntransport,), dtype=bool)

            h5f.create_dataset(self._kKey, (ntransport, 2))

            h5f.create_dataset(self._cputimeKey, (ntransport,))

            h5f.create_dataset(
                self._fluxKey, (ntransport, len(burnableIndexes), ngroups)
            )

            h5f.create_dataset(
                self._compKey, (ntransport, len(burnableIndexes), len(isotopes)),
            )

            isogroup = h5f.create_group(self._isotopeKey)
            zai = numpy.empty(len(isotopes), dtype=int)
            names = numpy.empty_like(zai, dtype=object)

            for ix, iso in enumerate(isotopes):
                zai[ix] = iso.zai
                names[ix] = iso.name

            isogroup["zais"] = zai
            isogroup["names"] = names.astype("S")

            materialgroup = h5f.create_group(self._matKey)
            mids = materialgroup.create_dataset(
                "ids", (len(burnableIndexes),), dtype=int
            )
            names = numpy.empty_like(mids, dtype=object)

            for ix, (matid, name) in enumerate(burnableIndexes):
                mids[ix] = matid
                names[ix] = name

            materialgroup["names"] = names.astype("S")

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
            tgroup["highFidelity"][timeindex] = not bool(timeStep.substep)

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


class HdfProcessor(Mapping):
    _EXPECTS = (0, 1)

    def __init__(
        self, fpOrGroup: typing.Union[str, pathlib.Path, h5py.File, h5py.Group]
    ):
        if isinstance(fpOrGroup, (str, pathlib.Path)):
            self._root = h5py.File(fpOrGroup, mode="r")
        elif isinstance(fpOrGroup, (h5py.File, h5py.Group)):
            self._root = fpOrGroup

        version = self._root.attrs.get("fileVersion")
        if version is None:
            raise KeyError(f"Could not find file version in {self._root}")
        elif tuple(version[:]) != self._EXPECTS:
            raise ValueError(
                f"Found {version[:]} in {self._root}, expected {self._EXPECTS}"
            )

        self.days = numpy.divide(self._root["time/time"], SECONDS_PER_DAY)

    def __len__(self) -> int:
        return len(self._root)

    def __getitem__(self, key) -> typing.Any:
        return self._root[key]

    def __iter__(self):
        return iter(self._root)

    def __contains__(self, key):
        return key in self._root

    def get(self, key, default=None) -> typing.Optional[typing.Any]:
        return self._root.get(key, default)

    def keys(self):
        return self._root.keys()

    def values(self):
        return self._root.values()

    def items(self):
        return self._root.items()

    @property
    def zais(self) -> numpy.ndarray:
        return self._root["isotopes/zais"][:]

    @property
    def keff(self) -> numpy.ndarray:
        return self._root["multiplicationFactor"][:]

    @property
    def hfFlags(self) -> numpy.ndarray:
        return self._root["time/highFidelity"][:]

    @property
    def fluxes(self) -> numpy.ndarray:
        return self._root["fluxes"][:]

    def sliceKeff(
        self, hfOnly: bool = True
    ) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        slicer = self.hfFlags if hfOnly else slice(None)
        return self.days[slicer], self.keff[slicer, :]

    def sliceFluxes(
        self, days: typing.Optional[typing.Union[float, typing.Iterable[float]]] = None
    ) -> numpy.ndarray:
        if days is None:
            dayslice = slice(None)
        elif isinstance(days, numbers.Real):
            dayslice = bisect.bisect_left(self.days, days)
            if dayslice == len(self.days) or days != self.days[dayslice]:
                raise IndexError(f"Day {days} not found")
        else:
            # Let numpy handle to searching
            reqs = numpy.asarray(days)
            if len(reqs.shape) != 1:
                raise ValueError("Days can only be 1D")
            if (reqs[:-1] - reqs[1:] > 0).any():
                raise ValueError("Days must be in increasing order, for now")
            dayslice = numpy.searchsorted(self.days, reqs)
            for ix, d in zip(dayslice, reqs):
                if ix == len(self.days) or self.days[ix] != d:
                    raise IndexError(f"Day {d} not found")
        # TODO Add group, material slicing
        return self["fluxes"][dayslice, :, 0]

    def getFissionMatrix(self, day: float) -> csr_matrix:
        ix = bisect.bisect_left(self.days, day)
        if ix == len(self.days) or self.days[ix] != day:
            raise IndexError(f"Day {day} not found")

        structure = self["fissionMatrix"].attrs.get("structure")
        if structure != "csr":
            raise ValueError("Expected csr matrix structure, not {structure}")

        group = self[f"fissionMatrix/{ix}"]

        return csr_matrix((group["data"], group["indices"], group["indptr"]))
