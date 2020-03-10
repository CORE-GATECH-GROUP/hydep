import numpy
import h5py
import scipy.sparse
import pytest
import hydep
from hydep.internal import TimeStep, TransportResult, CompBundle

h5store = pytest.importorskip("hydep.h5store")


@pytest.fixture
def result() -> TransportResult:
    """Return a TransportResult with arbitrary data"""
    N_GROUPS = 2
    N_BU_MATS = 8
    keff = [1.0, 1e-5]
    # Emulate multi-group data
    flux = numpy.arange(N_GROUPS * N_BU_MATS).reshape(N_BU_MATS, N_GROUPS)

    fmtx = scipy.sparse.rand(
        N_BU_MATS, N_BU_MATS, density=0.1, format="csr", random_state=123456
    )

    runtime = 10.1

    return TransportResult(flux, keff, runtime, fmtx=fmtx)


@pytest.fixture
def h5Destination(tmp_path):
    dest = (tmp_path / __file__).with_suffix(".h5")
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


def test_hdfStore(result, simpleChain, h5Destination):
    """Test that what goes in is what is written"""
    N_BU_MATS, N_GROUPS = result.flux.shape
    BU_INDEXES = [[x, f"mat {x}"] for x in range(N_BU_MATS)]

    # Emulate some depletion time
    START = TimeStep(0, 0, 0, 0)
    END = TimeStep(2, 1, 4, 10)

    store = h5store.HdfStore(filename=h5Destination,)
    assert store.fp.samefile(h5Destination)
    assert store.fp.is_absolute()

    assert store.VERSION[0] == 0, "Test not updated for current file version"

    # Need to add an additional step to account for zeroth time step
    store.beforeMain(
        END.coarse + 1, END.total + 1, N_GROUPS, tuple(simpleChain), BU_INDEXES,
    )

    store.postTransport(START, result)

    rng = numpy.random.default_rng(seed=123456)
    newDensities = [rng.random(len(simpleChain)) for _ in range(N_BU_MATS)]
    store.writeCompositions(END, CompBundle(tuple(simpleChain), newDensities))

    store.postTransport(END, result)

    with h5py.File(h5Destination, "r") as h5:
        assert tuple(h5.attrs["fileVersion"][:]) == store.VERSION
        assert tuple(h5.attrs["hydepVersion"][:]) == tuple(
            int(x) for x in hydep.__version__.split(".")[:3]
        )

        for ix, [matid, matname] in enumerate(BU_INDEXES):
            assert h5["/materials/ids"][ix] == matid
            assert h5["/materials/names"][ix].decode() == matname

        zais = h5["isotopes/zais"]
        names = h5["isotopes/names"]
        for ix, iso in enumerate(simpleChain):
            assert zais[ix] == iso.zai
            assert names[ix].decode() == iso.name

        assert len(simpleChain) == len(zais) == len(names)

        compareHdfStore(result, START, h5)
        compareHdfStore(result, END, h5)

        comps = h5["compositions"][END.total]

        for rx, rowDens in enumerate(newDensities):
            assert comps[rx] == pytest.approx(rowDens)

    # Test errors and warnings when creating a Store that may overwrite
    # existing files

    with pytest.raises(OSError):
        h5store.HdfStore(filename=h5Destination, existOkay=False)
