import copy

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
