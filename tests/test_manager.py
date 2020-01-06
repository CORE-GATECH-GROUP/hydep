"""
Tests for the depletion manager
"""
import copy

import pytest
import numpy
import hydep


def test_badmanager(simpleChain):
    """Test failure modes for manager construction"""

    with pytest.raises(TypeError):
        hydep.Manager(None, [1], 6e6)

    with pytest.raises(TypeError):
        hydep.Manager(simpleChain, [[1, 1], [2, 2]], [6e6])

    with pytest.raises(ValueError):
        hydep.Manager(simpleChain, [1], 0)

    with pytest.raises(ValueError):
        hydep.Manager(simpleChain, [1], [1, 2, 3])

    with pytest.raises(TypeError):
        hydep.Manager(simpleChain, [1, 1], [1, 0])

    with pytest.raises(TypeError):
        hydep.Manager(simpleChain, [1, 1], [1, -6e6])

    with pytest.raises(TypeError):
        hydep.Manager(simpleChain, [1, 1], {1, 2, 3})

    with pytest.raises(TypeError):
        hydep.Manager(simpleChain, [1], 6e6, numPreliminary=[1])

    with pytest.raises(ValueError):
        hydep.Manager(simpleChain, [1, 1, 1, 1], 6e6, numPreliminary=-1)

    with pytest.raises(ValueError):
        hydep.Manager(simpleChain, [1, 1, 1, 1], 6e6, numPreliminary=4)


def test_manager(toy2x2lattice, simpleChain):
    daysteps = numpy.array([5, 10])
    powers = [6e6, 10e6]

    man = hydep.Manager(simpleChain, daysteps, powers, numPreliminary=1)

    assert man.timesteps == pytest.approx(daysteps * 86400)

    for ix, (sec, power) in enumerate(man.preliminarySteps()):
        assert ix == 0
        assert sec == pytest.approx(daysteps[0] * 86400)
        assert power == powers[0]

    for ix, (sec, power) in enumerate(man.activeSteps()):
        assert ix == 0
        assert sec == pytest.approx(daysteps[1] * 86400)
        assert power == powers[1]

    toymodel = hydep.Model(toy2x2lattice)

    with pytest.raises(AttributeError, match=".*volume"):
        man.beforeMain(toymodel)

    fuelP = toymodel.root[0, 0]
    fuelM = fuelP.materials[0]
    assert isinstance(fuelM, hydep.BurnableMaterial)

    ORIG_FUEL_VOLUME = 100
    fuelM.volume = copy.copy(ORIG_FUEL_VOLUME)

    origBurnable = tuple(toymodel.findBurnableMaterials())
    assert len(origBurnable) == 1
    assert origBurnable[0] is fuelM

    man.beforeMain(toymodel)

    for ix, m in enumerate(toymodel.findBurnableMaterials()):
        assert ix < 1
        assert m.volume == pytest.approx(ORIG_FUEL_VOLUME)
        assert m.index == ix
        assert m is man.burnable[ix]
