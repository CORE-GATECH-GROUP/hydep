"""
Tests for the depletion manager
"""
import copy

import pytest
import numpy
import hydep
import hydep.constants
import hydep.internal


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


@pytest.fixture
def manager(simpleChain):
    daysteps = numpy.array([5, 10])
    powers = [6e6, 10e6]

    manager = hydep.Manager(simpleChain, daysteps, powers, numPreliminary=1)
    assert manager.timesteps == pytest.approx(daysteps * hydep.constants.SECONDS_PER_DAY)
    return manager


def test_manager(toy2x2lattice, manager):

    for ix, (sec, power) in enumerate(manager.preliminarySteps()):
        assert ix < manager.numPreliminary
        assert sec == manager.timesteps[ix]
        assert power == manager.powers[ix]

    numActive = len(manager.timesteps) - manager.numPreliminary
    for ix, (sec, power) in enumerate(manager.activeSteps()):
        assert ix < numActive
        assert sec == manager.timesteps[manager.numPreliminary + ix]
        assert power == manager.powers[manager.numPreliminary + ix]

    toymodel = hydep.Model(toy2x2lattice)

    with pytest.raises(AttributeError, match=".*volume"):
        manager.beforeMain(toymodel)

    fuelM = toymodel.root[0, 0].materials[0]
    assert isinstance(fuelM, hydep.BurnableMaterial)

    ORIG_FUEL_VOLUME = 100
    fuelM.volume = copy.copy(ORIG_FUEL_VOLUME)

    origBurnable = tuple(toymodel.root.findBurnableMaterials())
    assert len(origBurnable) == 1
    assert origBurnable[0] is fuelM

    # Create new burnable materials using Model interface
    # Update volumes along the way
    toymodel.differentiateBurnableMaterials(True)
    N_EXP_MATS = toymodel.root.size

    manager.beforeMain(toymodel)
    foundOrig = False

    for ix, m in enumerate(toymodel.root.findBurnableMaterials()):
        assert ix < N_EXP_MATS
        assert m.volume == pytest.approx(ORIG_FUEL_VOLUME / N_EXP_MATS)
        assert m.index == ix
        assert m is manager.burnable[ix]
        if m is fuelM:
            assert not foundOrig, "Found original fuel multiple times"
            foundOrig = True

    assert foundOrig, "Original fuel was not recovered"


@pytest.fixture
def microxs():
    zai = (922350, 922350, 922380, 922380, 541350)
    rxn = (18, 102, 18, 102, 102)
    mxs = numpy.array([[0.02], [0.06], [0.05], [0.07], [1e3]])
    return hydep.internal.MicroXsVector.fromLongFormVectors(
        zai, rxn, mxs, assumeSorted=False)


def test_oneGroupReactionRates(microxs, manager):
    times = [0, 100]
    incomingMicroXs = [[microxs], [microxs * 2]]
    expectedXsVector = microxs * 1.5

    VOLUME = 100
    burnedMaterial = hydep.BurnableMaterial("fuel", mdens=10, volume=VOLUME)
    water = hydep.Material("water", mdens=1)
    pin = hydep.Pin([0.05], [burnedMaterial], outer=water)
    model = hydep.Model(pin)

    manager.beforeMain(model)

    manager.setMicroXS(incomingMicroXs, times, polyorder=1)

    flux = numpy.array([[4E6]])

    # Reaction rates are expected to be one group values
    # Computed by integrating microscopic cross sections and flux / volume
    # over energy
    # Quantities should be a single vector
    expectedRates = expectedXsVector.mxs[:, 0] * flux[0, 0] / VOLUME

    rxns = manager.getReactionRatesAt(times[0] + 0.5 * (times[1] - times[0]), flux)

    assert len(rxns) == len(manager.burnable) == 1
    assert rxns[0].zai == expectedXsVector.zai
    assert rxns[0].rxns == expectedXsVector.rxns
    assert rxns[0].mxs == pytest.approx(expectedRates)
