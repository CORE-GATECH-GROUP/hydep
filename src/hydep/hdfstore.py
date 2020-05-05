""" Interface for writing results to HDF files

For the full format, see :ref:`hdf-format`.

"""
import numbers
import typing
import pathlib
from collections.abc import Mapping
import bisect
from enum import Enum

import numpy
import h5py
from scipy.sparse import csr_matrix

import hydep
from hydep.constants import SECONDS_PER_DAY
from .store import BaseStore


class HdfEnumKeys(Enum):
    """Enumeration that provides string encoding

    This allows using the instances directly, rather
    than having to make calls to their string values.
    """

    def encode(self, *args, **kwargs):
        """Return a encoded version of the enumeration value

        All arguments are passed directly to :meth:`str.encode`"""
        return self.value.encode(*args, **kwargs)


class HdfStrings(HdfEnumKeys):
    """Strings for root datasets or groups

    These can be used to interact with a file written
    by :class:`HdfStore` as keys to the root datasets
    or groups, e.g. ``hf[HdfStrings.FLUXES]`` would return
    the fluxes dataset.

    This class can also be paired with :class:`HdfSubStrings`
    to fetch second-tier data sets.

    Attributes
    ----------
    FLUXES : enum member
        Key to flux dataset
    KEFF : enum member
        Key to multiplication factor dataset
    ISOTOPES : enum member
        Key to isotope group
    COMPOSITIONS : enum member
        Key to compositions dataset
    CPU_TIMES : enum member
        Key to cpu time dataset
    MATERIALS : enum member
        Key to material group
    FISSION_MATRIX : enum member
        Key to fission matrix group
    CALENDAR : enum member
        Key to time step group

    """

    FLUXES = "fluxes"
    KEFF = "multiplicationFactor"
    ISOTOPES = "isotopes"
    COMPOSITIONS = "compositions"
    CPU_TIMES = "cpuTimes"
    MATERIALS = "materials"
    FISSION_MATRIX = "fissionMatrix"
    CALENDAR = "time"


class HdfSubStrings(HdfEnumKeys):
    """Strings for datasets or groups beyond the base group

    Attributes
    ----------
    MAT_IDS : enum member
        Key to material id dataset
    MAT_NAMES : enum member
        Key to material name dataset
    MAT_VOLS : enum member
        Key to material volume dataset
    CALENDAR_TIME : enum member
        Key to simulated time dataset
    CALENDAR_HF : enum member
        Key to high fidelity simulation flag dataset
    ISO_ZAI : enum member
        Key to isotope zai dataset
    ISO_NAMES : enum member
        Key to isotope name dataset

    """

    MAT_IDS = "ids"
    MAT_NAMES = "names"
    MAT_VOLS = "volumes"
    CALENDAR_TIME = "time"
    CALENDAR_HF = "highFidelity"
    ISO_ZAI = "zais"
    ISO_NAMES = "names"


class HdfAttrs(HdfEnumKeys):
    """Strings for populating the attributes dictionary

    Attributes
    ----------
    N_COARSE : enum member
        Key to number of high fidelity simulations
    N_TOTAL : enum member
        Key to total number of transport simulations
    N_ISOTOPES : enum member
        Key to number of isotopes tracked
    N_BMATS : enum member
        Key to number of burnable materials stored
    N_ENE_GROUPS : enum member
        Key to number of energy groups
    V_FORMAT : enum member
        Key to output file format number
    V_HYDEP : enum member
        Key to hydep version

    """

    N_COARSE = "coarseSteps"
    N_TOTAL = "totalSteps"
    N_ISOTOPES = "isotopes"
    N_BMATS = "burnableMaterials"
    N_ENE_GROUPS = "energyGroups"
    V_FORMAT = "fileVersion"
    V_HYDEP = "hydepVersion"


class HdfStore(BaseStore):
    """Write transport and depletion result to HDF files

    Format is described in :`hdf-format`

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
            h5f.attrs[HdfAttrs.V_FORMAT] = self.VERSION
            h5f.attrs[HdfAttrs.V_HYDEP] = tuple(
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
        burnableIndexes : iterable of [int, str, float]
            Each item is a 3-tuple of material id, name, and volume.
            Entries are ordered consistent to how the material are ordered
            are used across the sequence

        """
        with h5py.File(self._fp, "a") as h5f:
            for src, dest in (
                (nhf, HdfAttrs.N_COARSE),
                (ntransport, HdfAttrs.N_TOTAL),
                (len(isotopes), HdfAttrs.N_ISOTOPES),
                (len(burnableIndexes), HdfAttrs.N_BMATS),
                (ngroups, HdfAttrs.N_ENE_GROUPS),
            ):
                h5f.attrs[dest] = src

            tgroup = h5f.create_group(HdfStrings.CALENDAR)
            tgroup.create_dataset(HdfSubStrings.CALENDAR_TIME, (ntransport,))
            tgroup.create_dataset(HdfSubStrings.CALENDAR_HF, (ntransport,), dtype=bool)

            h5f.create_dataset(HdfStrings.KEFF, (ntransport, 2))

            h5f.create_dataset(HdfStrings.CPU_TIMES, (ntransport,))

            h5f.create_dataset(
                HdfStrings.FLUXES, (ntransport, len(burnableIndexes), ngroups)
            )

            h5f.create_dataset(
                HdfStrings.COMPOSITIONS,
                (ntransport, len(burnableIndexes), len(isotopes)),
            )

            isogroup = h5f.create_group(HdfStrings.ISOTOPES)
            zai = numpy.empty(len(isotopes), dtype=int)
            names = numpy.empty_like(zai, dtype=object)

            for ix, iso in enumerate(isotopes):
                zai[ix] = iso.zai
                names[ix] = iso.name

            isogroup[HdfSubStrings.ISO_ZAI] = zai
            isogroup[HdfSubStrings.ISO_NAMES] = names.astype("S")

            materialgroup = h5f.create_group(HdfStrings.MATERIALS)
            mids = materialgroup.create_dataset(
                HdfSubStrings.MAT_IDS, (len(burnableIndexes),), dtype=int
            )
            names = numpy.empty_like(mids, dtype=object)
            volumes = materialgroup.create_dataset_like(
                HdfSubStrings.MAT_VOLS, mids, dtype=numpy.float64
            )

            for ix, (matid, name, volume) in enumerate(burnableIndexes):
                mids[ix] = matid
                names[ix] = name
                volumes[ix] = volume

            materialgroup[HdfSubStrings.MAT_NAMES] = names.astype("S")

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
            tgroup = h5f[HdfStrings.CALENDAR]
            tgroup[HdfSubStrings.CALENDAR_TIME][timeindex] = timeStep.currentTime
            tgroup[HdfSubStrings.CALENDAR_HF][timeindex] = not bool(timeStep.substep)

            h5f[HdfStrings.KEFF][timeindex] = transportResult.keff

            h5f[HdfStrings.FLUXES][timeindex] = transportResult.flux

            cputime = transportResult.runTime
            if cputime is None:
                cputime = numpy.nan
            h5f[HdfStrings.CPU_TIMES][timeindex] = cputime

            fmtx = transportResult.fmtx
            if fmtx is not None:
                fGroup = h5f.get(HdfStrings.FISSION_MATRIX)
                if fGroup is None:
                    fGroup = h5f.create_group(HdfStrings.FISSION_MATRIX)
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
            h5f[HdfStrings.COMPOSITIONS][timeStep.total] = compBundle.densities


class HdfProcessor(Mapping):
    """Dictionary-like interface for HDF result files

    Properties like :attr:`zais` are generated at each
    call.

    Parameters
    ----------
    fpOrGroup : str or pathlib.Path or h5py.File or h5py.Group
        Either the name of the result file, an already opened
        HDF file or a group inside an opened HDF file

    Attributes
    ----------
    days : numpy.ndarray
        Points in calendar time for all provided values
    names : tuple of str
        Isotope names ordered consistent with :attr:`zai`.

    """

    _EXPECTS = (0, 1)

    def __init__(
        self, fpOrGroup: typing.Union[str, pathlib.Path, h5py.File, h5py.Group]
    ):
        if isinstance(fpOrGroup, (str, pathlib.Path)):
            self._root = h5py.File(fpOrGroup, mode="r")
        elif isinstance(fpOrGroup, (h5py.File, h5py.Group)):
            self._root = fpOrGroup
        else:
            raise TypeError(f"Type {type(fpOrGroup)} not supported")

        version = self._root.attrs.get("fileVersion")
        if version is None:
            raise KeyError(f"Could not find file version in {self._root}")
        elif tuple(version[:]) != self._EXPECTS:
            raise ValueError(
                f"Found {version[:]} in {self._root}, expected {self._EXPECTS}"
            )

        self.days = numpy.divide(self._root["time/time"], SECONDS_PER_DAY)
        self._names = None

    def __len__(self) -> int:
        return len(self._root)

    def __getitem__(self, key: str) -> typing.Union[h5py.Group, h5py.Dataset]:
        """Fetch a group or dataset directly from the file"""
        return self._root[key]

    def __iter__(self):
        return iter(self._root)

    def __contains__(self, key: str) -> bool:
        """Dictionary-like membership testing of ``key``"""
        return key in self._root

    def get(
        self, key: str, default: typing.Optional = None
    ) -> typing.Optional[typing.Any]:
        """Fetch a group or dataset from the file

        Parameters
        ----------
        key : str
            Name of the dataset or group of interest. Can contain
            multiple levels, e.g. ``"time/time"``
        default : object, optional
            Item to return if ``key`` is not found. Defaults to ``None``

        Returns
        -------
        object
            If ``key`` is found, will be either a :class:`h5py.Group`
            or :class:`h5py.Dataset`. Otherwise ``default`` is returned
        """
        return self._root.get(key, default)

    def keys(self):
        return self._root.keys()

    def values(self):
        return self._root.values()

    def items(self):
        return self._root.items()

    @property
    def zais(self) -> h5py.Dataset:
        """Ordered isotopic ZAI identifiers"""
        return self._root[HdfStrings.ISOTOPES][HdfSubStrings.ISO_ZAI]

    @property
    def names(self) -> typing.Tuple[str, ...]:
        # Stored on the processor to avoid decoding at every call
        if self._names is None:
            ds = self._root[HdfStrings.ISOTOPES][HdfSubStrings.ISO_NAMES]
            self._names = tuple((n.decode() for n in ds))
        return self._names

    @property
    def keff(self) -> h5py.Dataset:
        """Nx2 array with multiplication factor and absolute uncertainty

        Values will be provided for all transport solutions, even reduced
        order simulations that may not compute :math:`k_{eff}`. To obtain
        values at the high-fidelity points, see :meth:`getKeff`
        """
        return self._root[HdfStrings.KEFF]

    @property
    def hfFlags(self) -> h5py.Dataset:
        """Boolean vector indicating high fidelity (True) or reduced order solutions"""
        return self._root[HdfStrings.CALENDAR][HdfSubStrings.CALENDAR_HF][:]

    @property
    def fluxes(self) -> h5py.Dataset:
        """NxMxG array with fluxes in each burnable region

        Will be of shape ``(nTransport, nBurnable, nGroups)``

        """
        return self._root[HdfStrings.FLUXES]

    @property
    def compositions(self) -> h5py.Dataset:
        """NxMxI array with isotopic compositions

        Will be of shape ``(nTransport, nBurnable, nGroups)``

        """
        return self._root[HdfStrings.COMPOSITIONS]

    def getKeff(
        self, hfOnly: typing.Optional[bool] = True
    ) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        """Fetch the multiplication factor paired with the time in days

        Parameters
        ----------
        hfOnly : bool, optional
            Return the days and :math:`k` from high-fidelity solutions
            only [default]. Useful if the reduced order code does not
            compute / return :math:`k`.

        Returns
        -------
        days : numpy.ndarray
            Points in time [d] where :math:`k` has been evaluated
        keff : numpy.ndarray
            2D array with multiplication factor in the first column,
            absolute uncertainties in the second column

        """
        slicer = self.hfFlags if hfOnly else slice(None)
        return self.days[slicer], self.keff[slicer, :]

    def getFluxes(
        self, days: typing.Optional[typing.Union[float, typing.Iterable[float]]] = None
    ) -> numpy.ndarray:
        """Retrieve the flux at some or all time points

        Parameters
        ----------
        days : float or iterable of float, optional
            Specific day or days to obtain the flux. If multiple days
            are given, they must be in an increasing order

        Returns
        -------
        numpy.ndarray
            Fluxes at specified days. If ``day`` is a float, shape will
            be ``(nBurnable, nGroups)``. Otherwise, it will be
            ``(nDays, nBurnable, nGroups)`` where ``len(days) == nDays``

        Raises
        ------
        IndexError
            If ``days`` or an element of ``days`` was not found in
            :attr:`days`

        """
        # TODO Add group, material slicing
        if days is None:
            dayslice = slice(None)
        else:
            dayslice = self._getDaySlice(days)
        return self.fluxes[dayslice]

    def _getDaySlice(self, days: typing.Union[float, typing.Iterable[float]]):
        if isinstance(days, numbers.Real):
            dayslice = bisect.bisect_left(self.days, days)
            if dayslice == len(self.days) or days != self.days[dayslice]:
                raise IndexError(f"Day {days} not found")
            return dayslice

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
        return dayslice

    def getIsotopeIndexes(
        self,
        names: typing.Optional[typing.Union[str, typing.Iterable[str]]] = None,
        zais: typing.Optional[typing.Union[int, typing.Iterable[int]]] = None,
    ) -> typing.Union[int, numpy.ndarray]:
        """Return indices for specific isotopes

        Return type will match the input type. If a string or integer
        is passed, a single integer will be returned. If an iterable
        is passed, then a :class:`numpy.ndarray` will be returned.

        Parameters
        ----------
        names : str or iterable of str, optional
            Name or names to find in :attr:`names`
        zais : int or iterable of int, optional
            ZAI identifier(s) to find in :attr:`zais`

        Returns
        -------
        int or numpy.ndarray of int
            Indexes in :attr:`names` or :attr:`zais` that correspond
            to the input isotopes

        Raises
        ------
        ValueError
            If both ``names`` and ``zais`` are passed, or neither.
            Also raised in a name or zai not found

        """
        if names is not None:
            if zais is not None:
                raise ValueError("Need either names or zai, not both")
            if isinstance(names, str):
                try:
                    return self.names.index(names)
                except ValueError:
                    raise ValueError(f"Isotope {names} not found")
            return self._searchNames(names)

        if zais is not None:
            if isinstance(zais, numbers.Integral):
                ix = bisect.bisect_left(self.zais, zais)
                if ix == len(self.zais) or self.zais[ix] != zais:
                    raise ValueError(f"ZAI {zais} not found")
                return ix

            indices = numpy.searchsorted(self.zais, zais)
            for ix, z in zip(indices, zais):
                if ix == len(self.zais) or self.zais[ix] != z:
                    raise ValueError(f"ZAI {z} not found")
            return indices

        raise ValueError("Need either names or zai, not both")

    def _searchNames(self, names):
        indices = numpy.empty_like(names, dtype=int)

        for outindex, name in enumerate(names):
            try:
                ix = self.names.index(name)
            except ValueError:
                raise ValueError(f"Isotope {name} not found")
            indices[outindex] = ix

        return indices

    def getDensities(self, names=None, zais=None, days=None) -> numpy.ndarray:
        """Return atom densities for specific isotopes at specific times

        Parameters
        ----------
        names : str or iterable of str, optional
            Isotope name(s) e.g. ``"U235"``
        zais : int or iterable of int, optional
            Isotope ZAI identifier(s), e.g. ``922350``
        days : float or iterable of float, optional
            Retrieve densities for these points in time

        Returns
        -------
        numpy.ndarray
            Density in all materials at the requested times for the requested
            isotopes

        """
        if days is None:
            dayslice = slice(None)
        else:
            dayslice = self._getDaySlice(days)

        if names is None and zais is None:
            return self.compositions[dayslice]

        isoIndex = self.getIsotopeIndexes(names, zais)
        return self.compositions[dayslice][..., isoIndex]

    def getFissionMatrix(self, day: float) -> csr_matrix:
        """Retrieve the fission matrix for a given day

        Parameters
        ----------
        day : float
            Time in days that the matrix is requested

        Returns
        -------
        scipy.sparse.csr_matrix
            Fission matrix at time ``day``. Rows and columns correspond
            to unique burnable materials

        Raises
        ------
        KeyError
            If the fission matrix group is not defined
        IndexError
            If ``day`` was not found in :attr:`days`

        """
        fmtxGroup = self.get(HdfStrings.FISSION_MATRIX)
        if fmtxGroup is None:
            raise KeyError(
                "fissionMatrix group not found. Likely not included in simulation"
            )
        structure = fmtxGroup.attrs.get("structure")
        if structure != "csr":
            raise ValueError("Expected csr matrix structure, not {structure}")

        ix = bisect.bisect_left(self.days, day)
        if ix == len(self.days) or self.days[ix] != day:
            raise IndexError(f"Day {day} not found")

        group = fmtxGroup[str(ix)]

        return csr_matrix((group["data"], group["indices"], group["indptr"]))
