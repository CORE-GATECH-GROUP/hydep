import copy
import math

import numpy
import pytest
import hydep


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
    assert model.bounds.z is None


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
