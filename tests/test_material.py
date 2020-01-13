import numbers
import copy

import pytest
from hydep import Material, BurnableMaterial
import hydep.internal


@pytest.fixture
def isotopes():
    h1 = hydep.internal.getIsotope(name="H1")
    u235 = hydep.internal.getIsotope(name="U235")
    u238 = hydep.internal.getIsotope(zai=922380)
    o16 = hydep.internal.getIsotope(zai=(8, 16, 0))

    return {
        "H1": h1,
        "U235": u235,
        "U238": u238,
        "O16": o16,
    }


def test_material(isotopes):
    water = Material("water", mdens=1.0, H1=2, O16=1)

    assert len(water) == 2
    assert water[10010] == 2
    assert water[isotopes["O16"]] == 1
    assert water.mdens == 1.0
    water["O16"] = 0.9
    assert water[80160] == 0.9
    assert water.get("O16") == 0.9
    assert water.get(80160) == 0.9
    assert water.get(isotopes["O16"]) == 0.9

    water["O17"] = 5e-3
    assert len(water) == 3
    assert water[80170] == 5e-3

    assert water.adens is None
    assert water.temperature is None
    water.temperature = 600.0
    assert water.temperature == 600

    asStr = str(water)
    assert "g/cc" in asStr
    assert "at 600.0" in asStr

    with pytest.raises(AttributeError, match="Cannot set both atomic and mass"):
        water.adens = 1.0
    assert water.mdens == 1.0
    water.mdens = None
    water.adens = 2.0
    assert water.mdens is None
    assert water.adens == 2.0

    with pytest.raises(ValueError, match=r".*Material\.adens"):
        water.adens = -1

    with pytest.raises(TypeError, match=r".*Material\.adens"):
        water.adens = "1.0"

    assert water.id == 1


def test_burnableMaterial(isotopes):
    f = BurnableMaterial(
        "fuel", adens=2.68e-2, volume=10, temperature=900, U235=8e-4, O16=4.6e-4
    )

    assert len(f) == 2
    f[isotopes["U238"]] = 2.5e-2
    assert len(f) == 3
    assert f[922380] == 2.5e-2
    assert f["U235"] == 8e-4
    assert f[8, 16, 0] == 4.6e-4
    f["U235"] = 6e-4
    assert f[922350] == 6e-4
    assert f.volume == 10

    asStr = str(f)
    assert "atoms/b/cm" in asStr
    assert "cm^3" in asStr

    # Failures

    with pytest.raises(TypeError, match="Keys should be.*not.*float"):
        f[0.5] = 20
    with pytest.raises(ValueError, match=".*adens and mdens"):
        Material("bad", adens=1, mdens=1)

    with pytest.raises(AttributeError, match="Cannot set both atomic and mass"):
        f.mdens = 1.0
    assert f.adens == 2.68e-2
    f.adens = None
    f.mdens = 10.4
    assert f.adens is None
    assert f.mdens == 10.4

    with pytest.raises(ValueError, match=r".*BurnableMaterial\.mdens"):
        f.mdens = -1

    with pytest.raises(TypeError, match=r".*BurnableMaterial\.mdens"):
        f.mdens = "1.0"


@pytest.mark.parametrize("index", (1, "1", 1.0, -1))
def test_burnableIndex(index):
    f = hydep.BurnableMaterial("index tester")

    assert f.index is None

    if isinstance(index, numbers.Real) and index < 0:
        with pytest.raises(ValueError):
            f.index = index
        return

    f.index = index
    assert f.index == int(index)


def test_fixedBurnableIndex():
    """Test that BurnableMaterial.index is darn near fixed"""
    original = 2
    reference = copy.copy(original)

    mat = hydep.BurnableMaterial("index constancy")

    mat.index = original
    assert mat.index == reference

    with pytest.raises(AttributeError):
        mat.index *= 2

    assert mat.index == reference

    original *= 2

    assert original == 2 * reference
    assert mat.index == reference


def test_isotopes(isotopes):
    assert hydep.internal.getZaiFromName("U235") == (92, 235, 0)
    assert hydep.internal.getZaiFromName("H1") == (1, 1, 0)
    assert hydep.internal.getZaiFromName("Xe135_m1") == (54, 135, 1)
    assert hydep.internal.getZaiFromName("Pu239") == (94, 239, 0)

    u8 = isotopes["U238"]
    assert u8.zai == 922380
    assert u8.z == 92
    assert u8.a == 238
    assert u8.i == 0
    assert u8.triplet == (92, 238, 0)

    xeMeta = hydep.internal.getIsotope(name="Xe135_m1")
    assert xeMeta.triplet == (54, 135, 1)
    assert xeMeta.name == "Xe135_m1"

    shortU5 = hydep.internal.getIsotope(zai=(92, 235))
    assert shortU5 == isotopes["U235"]
    byTriplet = hydep.internal.getIsotope(zai=shortU5.triplet)
    assert byTriplet is shortU5

    u8Init0 = hydep.internal.Isotope("U238", 92, 238, 0)
    u8Init1 = hydep.internal.Isotope("U238", 92, 238)
    assert u8Init0 == u8Init1 == isotopes["U238"]

    assert u8 > shortU5
    assert u8 >= shortU5
    assert u8 >= u8Init0
    assert shortU5 < u8
    assert shortU5 <= u8
    assert shortU5 <= isotopes["U235"]


@pytest.mark.parametrize("attr", ["z", "a", "i"])
def test_badIsotope(attr):
    kwargs = {"z": 1, "a": 1, "i": 1}
    kwargs[attr] = 1.1

    with pytest.raises(TypeError, match=attr):
        hydep.internal.Isotope("not integer", **kwargs)

    kwargs[attr] = -1
    with pytest.raises(ValueError, match=attr):
        hydep.internal.Isotope("not integer", **kwargs)
