import numpy
import h5py
import scipy.sparse
import pytest
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
    assert tgroup["high fidelity"][index] != bool(timestep.substep)

    assert h5file["cpu times"][index] == pytest.approx(result.runTime)
    assert h5file["multiplication factor"][index] == pytest.approx(result.keff)
    assert h5file["fluxes"][index] == pytest.approx(result.flux)

    if result.fmtx is not None:
        fmtx = result.fmtx
        fgroup = h5file["fission matrix"]
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
    BU_INDEXES = [[x, "mat {}".format(x)] for x in range(N_BU_MATS)]

    # Emulate some depletion time
    START = TimeStep(0, 0, 0, 0)
    END = TimeStep(2, 1, 4, 10)

    # Need to add an additional step to account for zeroth time step
    store = HdfStore(
        END.coarse + 1,
        END.total + 1,
        len(simpleChain),
        N_BU_MATS,
        N_GROUPS,
        filename=h5Destination,
    )
    assert store.VERSION[0] == 0, "Test not updated for current file version"

    assert store.nCoarseSteps == END.coarse + 1
    assert store.nTotalSteps == END.total + 1
    assert store.nIsotopes == len(simpleChain)
    assert store.nGroups == N_GROUPS
    assert store.nBurnableMaterials == N_BU_MATS

    store.beforeMain(tuple(simpleChain), BU_INDEXES)

    store.postTransport(START, result)

    rng = numpy.random.default_rng(seed=123456)
    newDensities = [rng.random(len(simpleChain)) for _ in range(N_BU_MATS)]
    store.writeCompositions(END, CompBundle(tuple(simpleChain), newDensities))

    store.postTransport(END, result)

    with h5py.File(h5Destination, "r") as h5:
        for ix, [matid, matname] in enumerate(BU_INDEXES):
            grp = h5["materials/{}".format(ix)]
            assert grp.attrs["id"] == matid
            assert grp.attrs["name"] == matname
        assert h5.get("materials/{}".format(len(BU_INDEXES))) is None

        zais = []
        for ix, iso in enumerate(simpleChain):
            isoG = h5["isotopes/{}".format(ix)]
            zais.append(iso.zai)
            assert isoG.attrs["ZAI"] == iso.zai
            assert isoG.attrs["name"] == iso.name
        assert h5["isotopes/zais"][:] == pytest.approx(zais)
        assert h5.get("isotopes/{}".format(len(simpleChain))) is None

        compareHdfStore(result, START, h5)
        compareHdfStore(result, END, h5)

        comps = h5["compositions"][END.total]

        for rx, rowDens in enumerate(newDensities):
            assert comps[rx] == pytest.approx(rowDens)

    # Test errors and warnings when creating a Store that may overwrite
    # existing files

    with pytest.raises(OSError):
        HdfStore(1, 1, 1, 1, filename=h5Destination, existOkay=False)

    with pytest.warns(UserWarning):
        HdfStore(1, 1, 1, 1, filename=h5Destination, existOkay=True)
