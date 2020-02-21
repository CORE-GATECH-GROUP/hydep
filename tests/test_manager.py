"""
Tests for the depletion manager
"""
import copy
import collections

import pytest
import numpy
import hydep
import hydep.constants
import hydep.internal

SafeManagerArgs = collections.namedtuple(
    "SafeManagerArgs", "chain timesteps power divisions"
)


def test_managerConstruct(simpleChain):
    """Test manager construction"""

    safeargs = SafeManagerArgs(simpleChain, (1, 1), 6e6, 1)

    # Test that chain must be provided
    with pytest.raises(TypeError):
        hydep.Manager(None, [1], safeargs.power, safeargs.divisions)

    # Test that timesteps must be 1D
    with pytest.raises(TypeError):
        hydep.Manager(
            safeargs.chain, [[1, 1], [2, 2]], safeargs.power, safeargs.divisions
        )

    # Test that power must be > 0
    with pytest.raises(ValueError):
        hydep.Manager(safeargs.chain, safeargs.timesteps, 0, safeargs.divisions)

    with pytest.raises(ValueError):
        hydep.Manager(
            safeargs.chain, safeargs.timesteps, [safeargs.power, 0], safeargs.divisions,
        )

    with pytest.raises(ValueError):
        hydep.Manager(
            safeargs.chain,
            [1, 1],
            [safeargs.power, -safeargs.power],
            safeargs.divisions,
        )

    # Test that number of time steps == number of powers
    with pytest.raises(ValueError):
        hydep.Manager(safeargs.chain, [1], [safeargs.power] * 3, safeargs.divisions)

    # Test that power must be sequence (orderable)
    with pytest.raises(TypeError):
        hydep.Manager(
            safeargs.chain,
            [1, 1],
            {safeargs.power, 2 * safeargs.power},
            safeargs.divisions,
        )

    # Test that number of preliminary must be non-negative integer
    with pytest.raises(TypeError):
        hydep.Manager(
            safeargs.chain,
            safeargs.timesteps,
            safeargs.power,
            safeargs.divisions,
            numPreliminary=[1],
        )

    with pytest.raises(ValueError):
        hydep.Manager(
            safeargs.chain,
            [1, 1, 1, 1],
            safeargs.power,
            safeargs.divisions,
            numPreliminary=-1,
        )

    # Test that number of preliminary < number of provided time steps
    with pytest.raises(ValueError):
        hydep.Manager(
            safeargs.chain,
            [1, 1, 1, 1],
            safeargs.power,
            safeargs.divisions,
            numPreliminary=4,
        )


def test_managerSubsteps(simpleChain):
    ntimesteps = 3
    safeargs = SafeManagerArgs(simpleChain, (1,) * ntimesteps, 6e6, 1)

    m1 = hydep.Manager(*safeargs)

    assert len(m1.substeps) == ntimesteps
    assert all(s == safeargs.divisions for s in m1.substeps)

    scaled = list(safeargs.divisions * x for x in range(1, ntimesteps + 1))
    m2 = hydep.Manager(safeargs.chain, safeargs.timesteps, safeargs.power, scaled,)

    assert len(m2.substeps) == ntimesteps
    assert all(a == r for a, r in zip(m2.substeps, scaled)), (scaled, m2.substeps)

    # Test number of divisions must be positive integer
    with pytest.raises(ValueError):
        hydep.Manager(safeargs.chain, safeargs.timesteps, safeargs.power, 0)

    with pytest.raises(TypeError):
        hydep.Manager(safeargs.chain, safeargs.timesteps, safeargs.power, 1.5)

    with pytest.raises(ValueError):
        hydep.Manager(safeargs.chain, safeargs.timesteps, safeargs.power, -1)

    with pytest.raises(ValueError):
        hydep.Manager(safeargs.chain, safeargs.timesteps, safeargs.power, [-1])

    baddivisions = [safeargs.divisions] * (ntimesteps - 1) + [1.5]
    with pytest.raises(TypeError):
        hydep.Manager(safeargs.chain, safeargs.timesteps, safeargs.power, baddivisions)

    baddivisions[-1] = -1
    with pytest.raises(ValueError):
        hydep.Manager(safeargs.chain, safeargs.timesteps, safeargs.power, baddivisions)

    # Test that divisions must be orderable
    divset = {safeargs.divisions * x for x in range(1, ntimesteps + 1)}
    assert len(divset) == ntimesteps

    with pytest.raises(TypeError):
        hydep.Manager(safeargs.chain, safeargs.timesteps, safeargs.power, divset)

    # Test number of provided divisions equal to number of active steps
    with pytest.raises(ValueError):
        hydep.Manager(
            safeargs.chain,
            [1, 1, 1],
            safeargs.power,
            [safeargs.divisions],
            numPreliminary=1,
        )


@pytest.fixture
def manager(simpleChain):
    daysteps = numpy.array([5, 10])
    powers = [6e6, 10e6]
    divisions = 1

    manager = hydep.Manager(simpleChain, daysteps, powers, divisions, numPreliminary=1)
    assert manager.timesteps == pytest.approx(
        daysteps * hydep.constants.SECONDS_PER_DAY
    )
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
