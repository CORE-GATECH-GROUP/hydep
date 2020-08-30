import copy
import math

import numpy
import pytest
import hydep
from hydep.internal import Boundaries


def test_model(toy2x2lattice):

    with pytest.raises(TypeError):
        hydep.Model(None)

    with pytest.raises(TypeError):
        hydep.Model(hydep.BurnableMaterial("fuel", adens=1))

    model = hydep.Model(toy2x2lattice)

    assert len([model.root.findBurnableMaterials()]) == 1

    volume = 100
    referenceVol = copy.copy(volume)

    fuel = model.root[0, 0].materials[0]
    assert isinstance(fuel, hydep.BurnableMaterial)

    fuel.volume = volume

    model.differentiateBurnableMaterials(updateVolumes=True)
    assert volume == referenceVol

    N_EXPECTED = model.root.size
    foundOrig = False

    for ix, mat in enumerate(model.root.findBurnableMaterials()):
        assert ix < N_EXPECTED
        assert mat.volume == pytest.approx(referenceVol / N_EXPECTED)
        if mat is fuel:
            assert not foundOrig, "Found fuel multiple times"
            foundOrig = False

    assert model.bounds is None
    width = model.root.pitch * model.root.nx
    height = model.root.pitch * model.root.ny
    bounds = ((-width / 2, width), (-height / 2, height), None)
    model.bounds = bounds

    assert model.bounds.x == pytest.approx(bounds[0])
    assert model.bounds.y == pytest.approx(bounds[1])
    assert model.bounds.z == pytest.approx((-numpy.inf, numpy.inf))


@pytest.mark.parametrize("root", [True, False])
def test_boundsCheck(root):
    dummy = hydep.Material("dummy", adens=1)
    model = hydep.Model(hydep.InfiniteMaterial(dummy))

    assert model.bounds is None
    assert model.root.bounds is None
    assert not model.isBounded()

    if root:

        def setbounds(x=None, y=None, z=None):
            model.root.bounds = x, y, z

    else:

        def setbounds(x=None, y=None, z=None):
            model.bounds = x, y, z

    setbounds(None, None, None)
    assert not model.isBounded()
    for dim in {"x", "y", "z", "X", "Y", "Z", "all", "AlL"}:
        assert not model.isBounded(dim)

    for dim in {"x", "y", "z"}:
        setbounds(**{dim: (-1, 1)})
        assert model.isBounded(dim)
        setbounds(**{dim: None})

    for bounds in [
        (0, math.inf),
        (-math.inf, math.inf),
        (0, numpy.inf),
        (-math.inf, numpy.inf),
    ]:
        for dim in {"x", "y", "z"}:
            setbounds(**{dim: bounds})
            assert not model.isBounded(dim)
            assert not model.isBounded("all")
            setbounds(**{dim: None})

        setbounds(x=bounds, y=(1, 2), z=None)
        assert not model.isBounded()
        setbounds(x=(1, 2), y=bounds, z=None)
        assert not model.isBounded()

    # Check validity of 2D unbounded Z problems

    setbounds(x=(-1, 1), y=(-1, 1), z=None)
    assert model.isBounded()

    setbounds(x=numpy.arange(2), y=[-0.6, 0.6], z=[0, 360])
    assert model.isBounded()
    assert model.isBounded("x")
    assert model.isBounded("y")
    assert model.isBounded("z")


@pytest.fixture
def simpleModel():
    univ = hydep.InfiniteMaterial(hydep.Material("inf", adens=1))
    return hydep.Model(univ)


def checkbounds(actual, expected):
    assert actual.x == expected.x
    assert actual.y == expected.y
    assert actual.z == expected.z


@pytest.mark.parametrize("presetBounds", [True, False])
def test_axialSymmetry(simpleModel, presetBounds):
    bounds = Boundaries((-1, 1), (-1, 1), (0, 1))
    simpleModel.root.bounds = bounds
    assert simpleModel.bounds is None

    if presetBounds:
        simpleModel.bounds = bounds

    assert not simpleModel.axialSymmetry

    simpleModel.applyAxialSymmetry()

    assert simpleModel.axialSymmetry
    checkbounds(
        simpleModel.bounds,
        Boundaries(bounds.x, bounds.y, (0, bounds.z.upper))
    )
    checkbounds(simpleModel.root.bounds, bounds)


def test_symmetryFailure(simpleModel):
    # No boundaries are declared at all
    with pytest.raises(hydep.GeometryError, match=".*root universe.*unbounded"):
        simpleModel.applyAxialSymmetry()

    assert not simpleModel.axialSymmetry

    # Unbounded in z direction
    for z in [None, (0, numpy.inf), (-numpy.inf, 1)]:
        simpleModel.bounds = ((-1, 1), (-1, 1), z)
        with pytest.raises(hydep.GeometryError, match=".*unbounded.* z "):
            simpleModel.applyAxialSymmetry()

        assert not simpleModel.axialSymmetry

    simpleModel.bounds = (-1, 1), (-1, 1), (1, 2)
    with pytest.raises(hydep.GeometryError, match=".*z boundary"):
        simpleModel.applyAxialSymmetry()
    assert not simpleModel.axialSymmetry

    # origin not contained in xy plane
    for coords in [(1, 2), (-numpy.inf, -1)]:
        simpleModel.bounds = coords, (-1, 1), (0, 1)
        with pytest.raises(hydep.GeometryError, match=".*xy"):
            simpleModel.applyAxialSymmetry()

        assert not simpleModel.axialSymmetry


def test_symmetryTypes():
    assert hydep.Symmetry.fromStr("none") is hydep.Symmetry.NONE
    assert hydep.Symmetry.fromInt(2) is hydep.Symmetry.HALF
    with pytest.raises(ValueError):
        hydep.Symmetry.fromStr("failure")

    # Really any value > 1 could work, as this would imply
    # 360 / N slices. But use a ridiculous number here
    with pytest.raises(ValueError):
        hydep.Symmetry.fromInt(500)
