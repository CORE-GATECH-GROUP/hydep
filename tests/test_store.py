import random

import numpy
import pytest
h5py = pytest.importorskip("h5py")
import scipy.sparse
import hydep
from hydep.internal import TimeStep, TransportResult, CompBundle
import hydep.hdfstore

N_GROUPS = 2
N_BU_MATS = 2
VOLUME = 0.12345
BU_INDEXES = [[x, f"mat {x}", VOLUME] for x in range(N_BU_MATS)]

# Emulate some depletion time
START = TimeStep(0, 0, 0, 0)
END = TimeStep(2, 1, 4, 10 * hydep.constants.SECONDS_PER_DAY)


@pytest.fixture(scope="module")
def result() -> TransportResult:
    """Return a TransportResult with arbitrary data"""
    keff = [1.0, 1e-5]
    # Emulate multi-group data
    flux = numpy.arange(N_GROUPS * N_BU_MATS).reshape(N_BU_MATS, N_GROUPS)

    fmtx = scipy.sparse.eye(N_BU_MATS, N_BU_MATS, format="csr")
    assert fmtx.nnz

    runtime = 10.1

    return TransportResult(flux, keff, runtime, fmtx=fmtx)


@pytest.fixture(scope="module")
def compositions(simpleChain):
    rng = numpy.random.default_rng(seed=123456)
    return CompBundle(tuple(simpleChain), rng.random((N_BU_MATS, len(simpleChain))))


@pytest.fixture
def h5Destination(tmp_path, result, compositions, simpleChain):
    dest = (tmp_path / __file__).with_suffix(".h5")

    store = hydep.hdfstore.HdfStore(filename=dest)
    assert store.fp.samefile(dest)
    assert store.fp.is_absolute()

    assert store.VERSION[0] == 0, "Test not updated for current file version"

    # Need to add an additional step to account for zeroth time step
    store.beforeMain(
        END.coarse + 1, END.total + 1, N_GROUPS, tuple(simpleChain), BU_INDEXES,
    )
    store.postTransport(START, result)

    store.writeCompositions(END, compositions)

    store.postTransport(END, result)
    yield dest
    dest.unlink()


def compareHdfStore(result, timestep, h5file):
    # Use pytest.approx for floating point comparisons
    index = timestep.total

    tgroup = h5file["time"]
    assert tgroup["time"][index] == pytest.approx(timestep.currentTime)
    assert tgroup["highFidelity"][index] != bool(timestep.substep)

    assert h5file["cpuTimes"][index] == pytest.approx(result.runTime)
    assert h5file["multiplicationFactor"][index] == pytest.approx(result.keff)
    assert h5file["fluxes"][index] == pytest.approx(result.flux)

    if result.fmtx is not None:
        fmtx = result.fmtx
        fgroup = h5file["fissionMatrix"]
        shape = fgroup.attrs["shape"]
        assert len(shape) == 2 and shape[0] == shape[1], shape

        datag = fgroup[str(index)]
        assert datag.attrs["nnz"] == fmtx.nnz

        for attr in {"data", "indices", "indptr"}:
            ref = getattr(fmtx, attr)
            actual = datag[attr][:]
            assert actual == pytest.approx(ref), attr


def test_hdfStore(result, simpleChain, h5Destination, compositions):
    """Test that what goes in is what is written"""

    with h5py.File(h5Destination, "r") as h5:
        assert tuple(h5.attrs["fileVersion"][:]) == (0, 1)
        assert tuple(h5.attrs["hydepVersion"][:]) == tuple(
            int(x) for x in hydep.__version__.split(".")[:3]
        )

        for ix, [matid, matname, volume] in enumerate(BU_INDEXES):
            assert h5["/materials/ids"][ix] == matid
            assert h5["/materials/names"][ix].decode() == matname
            assert h5["/materials/volumes"][ix] == volume

        zais = h5["isotopes/zais"]
        names = h5["isotopes/names"]
        for ix, iso in enumerate(simpleChain):
            assert zais[ix] == iso.zai
            assert names[ix].decode() == iso.name

        assert len(simpleChain) == len(zais) == len(names)

        compareHdfStore(result, START, h5)
        compareHdfStore(result, END, h5)

        comps = h5["compositions"][END.total]

        for rx, rowDens in enumerate(compositions.densities):
            assert comps[rx] == pytest.approx(rowDens)

    # Test errors and warnings when creating a Store that may overwrite
    # existing files

    with pytest.raises(OSError):
        hydep.hdfstore.HdfStore(filename=h5Destination, existOkay=False)


def test_hdfProcessor(result, simpleChain, compositions, h5Destination):

    processor = hydep.hdfstore.HdfProcessor(h5Destination)

    assert processor.days[START.total] == pytest.approx(
        START.currentTime / hydep.constants.SECONDS_PER_DAY
    )
    assert processor.days[END.total] == pytest.approx(
        END.currentTime / hydep.constants.SECONDS_PER_DAY
    )

    assert processor.keff.shape == (processor.days.size, 2)
    assert processor.keff[START.total] == pytest.approx(result.keff)
    assert processor.keff[END.total] == pytest.approx(result.keff)
    hfDays, hfKeff = processor.getKeff(hfOnly=True)
    flags = processor.hfFlags[:]
    assert hfDays == processor.days[flags]
    assert hfKeff == pytest.approx(processor.keff[flags, :])

    fullDays, fullKeff = processor.getKeff(hfOnly=False)
    assert fullDays == pytest.approx(processor.days)
    assert fullKeff == pytest.approx(processor.keff[:])

    assert processor.compositions.shape == (processor.days.size, N_BU_MATS, len(simpleChain))
    assert processor.compositions[END.total] == pytest.approx(compositions.densities)

    assert processor.fluxes.shape == (processor.days.size, N_BU_MATS, N_GROUPS)
    assert processor.fluxes[START.total] == pytest.approx(result.flux)
    assert processor.fluxes[END.total] == pytest.approx(result.flux)
    assert processor.getFluxes(processor.days[START.total]) == pytest.approx(result.flux)

    fluxes = processor.getFluxes(processor.days[flags])
    assert fluxes == pytest.approx(processor.fluxes[flags, :])
    assert processor.getFluxes() == pytest.approx(processor.fluxes[:])

    assert len(processor.names) == len(simpleChain)
    assert len(processor.zais) == len(simpleChain)
    for iso, name, zai in zip(simpleChain, processor.names, processor.zais):
        assert name == iso.name
        assert zai == iso.zai
        nix = processor.getIsotopeIndexes(names=name)
        zix = processor.getIsotopeIndexes(zais=zai)
        assert processor.names[nix] == name
        assert processor.zais[zix] == zai
        assert nix == zix

    randomNames = random.sample(processor.names, k=10)
    indices = processor.getIsotopeIndexes(names=randomNames)
    assert len(indices) == 10
    for ix, name in zip(indices, randomNames):
        assert processor.names[ix] == name

    assert processor.volumes.shape == (N_BU_MATS, )
    assert numpy.allclose(processor.volumes, VOLUME)

    randomDens = processor.getDensities(names=randomNames)
    assert randomDens == pytest.approx(processor.compositions[:][..., indices])
    assert processor.getDensities(names=randomNames[0]) == pytest.approx(
            processor.compositions[..., indices[0]])

    with pytest.raises(ValueError, match=".*bad name"):
        processor.getIsotopeIndexes(names="bad name")

    with pytest.raises(ValueError, match=".*bad name"):
        processor.getIsotopeIndexes(names=randomNames + ["bad name"])

    # Numpy arrays and h5py datasets are not sequences
    # Need to convert to list to obtain a random sample
    randomZais = random.sample(list(processor.zais), k=10)
    indices = processor.getIsotopeIndexes(zais=randomZais)
    assert len(indices) == 10
    for ix, zai in zip(indices, randomZais):
        assert processor.zais[ix] == zai

    randomDens = processor.getDensities(zais=randomZais)
    assert randomDens == pytest.approx(processor.compositions[:][..., indices])
    assert processor.getDensities(zais=randomZais[0]) == pytest.approx(
            processor.compositions[..., indices[0]])

    assert processor.getDensities() == pytest.approx(processor.compositions[:])
    assert processor.getDensities(
        days=processor.days[0], names=processor.names[0]) == pytest.approx(
            processor.compositions[0, :, 0][:])
    assert processor.getDensities(
        days=[processor.days[0], processor.days[-1]], names=processor.names[:5]) == pytest.approx(
            processor.compositions[(0, END.total), :, :5])

    with pytest.raises(ValueError, match=".*0"):
        processor.getIsotopeIndexes(zais=0)

    with pytest.raises(ValueError, match=".*0"):
        processor.getIsotopeIndexes(zais=randomZais + [0])

    with pytest.raises(ValueError):
        processor.getIsotopeIndexes()
    with pytest.raises(ValueError):
        processor.getIsotopeIndexes(names=randomNames, zais=randomZais)

    for time in [START, END]:
        fmtx = processor.getFissionMatrix(processor.days[time.total])
        assert fmtx.data == pytest.approx(result.fmtx.data)
        assert fmtx.indices == pytest.approx(result.fmtx.indices)
        assert fmtx.indptr == pytest.approx(result.fmtx.indptr)

    # Compare group and dataset references
    for key, item in processor.items():
        assert key in processor
        assert processor[key].id == item.id
        assert processor.get(key).id == item.id

    assert "fake key" not in processor
    with pytest.raises(KeyError):
        processor["fake key"]
    assert processor.get("fake key") is None


def test_hdfenums():
    RootNames = hydep.hdfstore.HdfStrings
    SecondNames = hydep.hdfstore.HdfSubStrings

    bypathop = RootNames.CALENDAR / SecondNames.CALENDAR_TIME
    expected = "/".join([o.value for o in [RootNames.CALENDAR, SecondNames.CALENDAR_TIME]])
    assert bypathop == expected

    bydig = RootNames.CALENDAR.dig(SecondNames.CALENDAR_TIME)
    assert bydig == expected

    bydig = RootNames.CALENDAR.dig(SecondNames.CALENDAR_TIME, "foo")
    assert bydig == expected + "/foo"
