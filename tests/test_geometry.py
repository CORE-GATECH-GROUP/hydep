from collections import namedtuple

import numpy
import pytest
import hydep

PinBundle = namedtuple("PinFixture", ["name", "radii", "materials", "findOrder"])


@pytest.fixture
def materials():
    # TODO Global fixture?
    fuel = hydep.BurnableMaterial(
        "fuel", mdens=10.4, U235=8.0e-4, U238=2.5e-2, O16=4.6e-4
    )
    water = hydep.Material("water", mdens=1.0, H1=4.7e-2, O16=2.4e-2)

    clad = hydep.Material("clad", mdens=6.5, Zr90=4.32e-2)

    return {mat.name: mat for mat in [fuel, water, clad]}


@pytest.fixture
def bundles():
    EXP_FP = PinBundle(
        "fuel",
        (0.4005, 0.42, numpy.inf),
        ["fuel", "clad", "water"],
        ["fuel", "clad", "water"],
    )

    EXP_GT = PinBundle(
        "guide", (0.5, 0.55, numpy.inf), ["water", "clad", "water"], ["water", "clad"],
    )

    EXP_DIV = PinBundle(
        "segmented",
        (0.2, 0.4005, 0.45, numpy.inf),
        ["fuel", "fuel", "clad", "water"],
        ["fuel", "clad", "water"],
    )
    return {b.name: b for b in [EXP_FP, EXP_GT, EXP_DIV]}


def pinFromBundle(materials, bundle):
    return hydep.Pin(
        bundle.radii[:-1],
        [materials[k] for k in bundle.materials[:-1]],
        materials[bundle.materials[-1]],
        name=bundle.name,
    )


@pytest.fixture
def pins(materials, bundles):
    fuelPin = pinFromBundle(materials, bundles["fuel"])
    guidePin = pinFromBundle(materials, bundles["guide"])
    divFuel = pinFromBundle(materials, bundles["segmented"])
    assert fuelPin.id == 1
    assert guidePin.id == 2
    assert divFuel.id == 3
    return {p.name: p for p in [fuelPin, guidePin, divFuel]}


@pytest.mark.parametrize("key", ("fuel", "guide", "segmented"))
def test_pin(materials, pins, bundles, key):
    pin = pins[key]
    expectedBundle = bundles[key]

    assert pin.name == expectedBundle.name
    assert len(pin) == len(expectedBundle.materials)
    assert pin.radii == expectedBundle.radii[:-1]
    for actualMat, matName in zip(pin.materials, expectedBundle.materials[:-1]):
        assert actualMat is materials[matName]
    assert pin.outer is materials[expectedBundle.materials[-1]]

    found = list(pin.findMaterials())
    assert len(found) == len(expectedBundle.findOrder)
    for f, name in zip(found, expectedBundle.findOrder):
        assert f is materials[name]

    if "fuel" in expectedBundle.materials:
        expectedBurnable = {"fuel": expectedBundle.materials.count("fuel")}
    else:
        expectedBurnable = {}

    actualBurnable = {m.name: c for m, c in pin.countBurnableMaterials().values()}
    assert len(actualBurnable) == len(expectedBurnable)
    assert sum(actualBurnable.values()) == expectedBurnable.get("fuel", 0)

    assert pin.boundaries() is None


def test_pinDiffBurnedMaterials(materials, pins):
    # Differentiation of burnable materials
    # With a single burnable material, return the same Pin instance
    # if this is the first encounter
    memo = set()
    testpin = pins["fuel"]
    assert testpin.differentiateBurnableMaterials(memo) is testpin
    assert len(memo) == 2
    assert id(testpin) in memo
    assert id(materials["fuel"]) in memo

    # New fuel has been added to the memo, so it will be cloned on the
    # next pass, and a new pin will be returned
    cloned = testpin.differentiateBurnableMaterials(memo)
    assert isinstance(cloned, testpin.__class__)
    assert cloned is not testpin
    assert cloned.id != testpin.id
    assert cloned.radii == testpin.radii

    # Check that all materials except fuel are identical
    found = list(cloned.findMaterials())
    assert len(found) == len(materials)

    for mat in found:
        if isinstance(mat, hydep.BurnableMaterial):
            assert mat is not materials["fuel"]
        else:
            assert mat is materials[mat.name]

    # Run again on test pin assuming fuel material has been discovered
    # elsewhere -> modify this pin in place
    memo = {id(materials["fuel"])}
    assert testpin.differentiateBurnableMaterials(memo) is testpin
    assert len(memo) == 3
    assert id(materials["fuel"]) in memo
    assert id(testpin) in memo
    for mat in testpin.materials:
        if isinstance(mat, hydep.BurnableMaterial):
            assert mat is not materials["fuel"]
            assert mat.id != materials["fuel"].id
            assert id(mat) in memo
        else:
            assert mat is materials[mat.name]

    # Re-run and obtain new pin class just to be sure
    assert testpin.differentiateBurnableMaterials(memo) is not testpin

    # Segmented pin will have a unique material created
    # on first pass, but will not be cloned
    memo = set()
    testpin = pins["segmented"]
    assert testpin.differentiateBurnableMaterials(memo) is testpin
    segmats = list(testpin.findMaterials())
    assert len(segmats) == len(materials) + 1
    segbumats = list(testpin.findBurnableMaterials())
    assert len(segbumats) == 2
    assert id(materials["fuel"]) in set(
        id(m) for m in segbumats
    ), "Original fuel not found"


@pytest.fixture
def pinArray(pins):
    return [
        [pins["fuel"]] * 3,
        [pins["fuel"], pins["guide"], pins["fuel"]],
        [pins["fuel"]] * 3,
    ]


def test_cartesianLattice(materials, pins, pinArray):
    lat = hydep.CartesianLattice(3, 3, pitch=1.26)
    assert lat.id == len(pins) + 1
    hwidth = 1.5 * lat.pitch

    # Act on "empty" lattice
    assert len(lat) == 3

    for _item in lat:
        raise RuntimeError("Iteration on empty array should not be supported")

    with pytest.raises(AttributeError, match="Array not set"):
        lat[0, 1]

    with pytest.raises(AttributeError, match="Array not set"):
        list(lat.findMaterials())

    with pytest.raises(AttributeError, match="Array not set"):
        lat.countBurnableMaterials()

    lat[0, 1] = pins["guide"]
    assert lat[0, 1] is pins["guide"]
    assert lat[0, 0] is None  # default numpy empty object

    lat.array = pinArray
    assert lat.nx == 3
    assert lat.ny == 3
    assert lat.array.shape == lat.shape == (3, 3)
    assert lat.pitch == 1.26
    assert lat.array.size == lat.size == 9
    assert lat.array.dtype == object
    assert lat.boundaries() == ((-hwidth, hwidth), (-hwidth, hwidth), None)

    assert lat[0, 0] is pins["fuel"]
    assert lat[1, 1] is pins["guide"]

    for rx, row in enumerate(lat):
        assert row.size == 3
        for cx, cpin in enumerate(row):
            assert cpin is pinArray[rx][cx]

    mats = list(lat.findMaterials())
    assert len(mats) == len(materials)

    for mat in mats:
        assert mat is materials[mat.name]

    burnable = list(lat.findBurnableMaterials())
    assert len(burnable) == 1
    assert burnable[0] is materials["fuel"]

    burnCount = lat.countBurnableMaterials()
    assert len(burnCount) == 1
    _hid, (mat, count) = burnCount.popitem()

    assert mat is materials["fuel"]
    assert count == 8

    lat[1, 1] = pins["segmented"]
    assert lat[1, 1] is pins["segmented"]

    burnCount = lat.countBurnableMaterials()
    assert len(burnCount) == 1
    _hid, (mat, count) = burnCount.popitem()

    assert mat is materials["fuel"]
    assert count == 10

    for l, a in zip(lat.flat, lat.array.flat):
        assert l is a


def test_diffBuLattice(materials, pins, pinArray):
    lattice = hydep.CartesianLattice(
        3, 3, pitch=1.26, array=pinArray, name="cloned lattice"
    )
    assert len(list(lattice.findMaterials())) == len(materials)
    assert len(list(lattice.findBurnableMaterials())) == 1

    # Nine total pins in pin array
    # One is guide tube -> 8 fuel pins
    # All are originally the same fuel pin -> one will not be cloned
    NEW_PINS = 7

    memo = set()
    assert lattice.differentiateBurnableMaterials(memo) is lattice
    assert id(lattice) in memo

    foundguide = False
    for ix, p in enumerate(lattice.flat):
        if p is pins["guide"]:
            foundguide = True
            continue
        if ix == 0:
            assert p is pins["fuel"]
        else:
            assert p is not pins["fuel"]
    assert foundguide

    assert len(list(lattice.findMaterials())) == len(materials) + NEW_PINS
    assert len(list(lattice.findBurnableMaterials())) == 1 + NEW_PINS

    # Repeat again to ensure that a new array is created
    newlat = lattice.differentiateBurnableMaterials(memo)
    assert newlat is not lattice
    assert newlat.shape == lattice.shape
    assert newlat.pitch == lattice.pitch
    assert newlat.name == lattice.name
    assert newlat.id != lattice.id

    foundguide = False
    for p in newlat.flat:
        if p is pins["guide"]:
            foundguide = True
        else:
            assert p is not pins["fuel"]
    assert foundguide


def test_latticeStack(materials, pins, pinArray):
    xylattice = hydep.CartesianLattice(3, 3, pitch=1.26, array=pinArray)
    assert xylattice.id == len(pins) + 1
    stack = hydep.LatticeStack(1, heights=(0, 1), items=[xylattice])
    assert stack.id == xylattice.id + 1

    hwidth = xylattice.nx * xylattice.pitch * 0.5

    assert len(stack) == 1
    assert stack.items[0] is stack[0] is xylattice

    for ix, item in enumerate(stack):
        assert ix == 0, "Over iterated"
        assert item is xylattice

    numpy.testing.assert_equal(stack.heights, (0, 1))

    found = list(stack.findMaterials())
    assert len(found) == len(set(m.name for m in found))
    assert len(found) == len(materials)

    burnable = stack.countBurnableMaterials()
    assert len(burnable) == 1
    _hid, (mat, count) = burnable.popitem()
    assert mat is materials["fuel"]
    assert count == 8

    assert stack.boundaries() == ((-hwidth, hwidth), (-hwidth, hwidth), (0, 1))


def test_minicore(materials, pins, pinArray):
    PINS_PRIMARY = 8
    PINS_NEIGHBOR = 5

    primary = hydep.CartesianLattice(3, 3, pitch=1.26, array=pinArray)
    neighbor = hydep.CartesianLattice(3, 3, pitch=1.26)
    neighbor.array = [
        [pins["fuel"], pins["guide"], pins["fuel"]],
        [pins["guide"], pins["fuel"], pins["guide"]],
        [pins["fuel"], pins["guide"], pins["fuel"]],
    ]

    stack = hydep.LatticeStack(2, heights=(0, 180, 360), items=[primary, neighbor])

    minicore = hydep.CartesianLattice(
        2, 1, pitch=3 * 1.26, array=[[stack, stack]], name="minicore"
    )

    assert minicore.size == 2
    assert minicore.shape == (1, 2)
    assert minicore[0, 0] is minicore[0, 1] is stack
    assert minicore[0, 0][0] is primary
    assert minicore[0, 0][1] is neighbor

    found = list(minicore.findMaterials())
    assert len(found) == len(materials)
    assert len(set(id(m) for m in found)) == len(materials)
    for m in found:
        assert m is materials[m.name]

    burnable = minicore.countBurnableMaterials()
    assert len(burnable) == 1
    _hid, (mat, count) = burnable.popitem()
    assert mat is materials["fuel"]
    assert count == 2 * (PINS_PRIMARY + PINS_NEIGHBOR)

    assert minicore.differentiateBurnableMaterials() is minicore
    assert minicore[0, 0] is stack
    assert minicore[0, 0][0] is primary
    assert minicore[0, 0][1] is neighbor

    assert minicore[0, 1] is not stack

    # Count new materials
    newburnable = minicore.countBurnableMaterials()
    assert len(newburnable) == count
    for _mat, count in newburnable.values():
        assert count == 1
