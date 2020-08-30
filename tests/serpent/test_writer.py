import io
import pathlib

import numpy
import pytest
import hydep
from hydep.internal import (
    CompBundle,
    TimeStep,
    compBundleFromMaterials,
    Boundaries,
)
import hydep.serpent
from hydep.serpent.utils import Library

from tests import strcompare, filecompare
from tests.regressions import config


REF_FUEL = """% fuel32
mat 1 -10.3400000 burn 1
8016.09c 4.602600000E-02
8017.09c 1.753300000E-05
92234.09c 5.995900000E-06
92235.09c 7.462900000E-04
92238.09c 2.231700000E-02
"""

REF_WATER = """% water
mat 4 -0.7405000 moder HinH2O_600.00 1001
1001.06c 4.945800000E-02
1002.06c 5.688300000E-06
5011.06c 3.221000000E-05
8016.06c 2.467200000E-02
8017.06c 9.398100000E-06
"""

REF_DEFS = {"fuel32": REF_FUEL, "water": REF_WATER}


@pytest.mark.serpent
@pytest.mark.parametrize("material", ("fuel32", "water"))
def test_materials(beavrsMaterials, material):
    mat = beavrsMaterials[material]
    reference = REF_DEFS[material]
    stream = io.StringIO()
    writer = hydep.serpent.SerpentWriter()

    writer.writemat(stream, mat)

    written = stream.getvalue()
    assert strcompare(reference, written)


@pytest.mark.serpent
def test_writePlainPin(beavrsInstrumentTube):
    writer = hydep.serpent.SerpentWriter()
    stream = io.StringIO()

    writer.writeUniverse(stream, beavrsInstrumentTube, {})
    reference = """% {}
pin p{}
2 0.4368800
3 0.4838700
4 0.5613400
3 0.6019800
4

""".format(
        beavrsInstrumentTube.name, beavrsInstrumentTube.id
    )
    written = stream.getvalue()
    assert strcompare(reference, written)


@pytest.mark.serpent
def test_writeFuelPin(beavrsFuelPin):
    writer = hydep.serpent.SerpentWriter()
    stream = io.StringIO()

    writer.writeUniverse(stream, beavrsFuelPin, {})
    reference = """surf 1_r0 inf
cell 1_r0 1 1 -1_r0
pin p1
fill 1 0.3921800
5 0.4000500
3 0.4572000
4

"""
    written = stream.getvalue()
    assert strcompare(reference, written)


@pytest.mark.serpent
def test_writeInfMaterial(beavrsMaterials):
    reference = """% Infinite region filled with {name}
surf inf1 inf
cell inf1 inf1 {} -inf1
"""
    universe = hydep.InfiniteMaterial(beavrsMaterials["ss304"])
    reference = """% Infinite region filled with {name}
surf inf{uid} inf
cell inf{uid} inf{uid} {mid} -inf{uid}
""".format(
        name=universe.material.name, mid=universe.material.id, uid=universe.id
    )

    writer = hydep.serpent.SerpentWriter()
    stream = io.StringIO()

    writer.writeUniverse(stream, universe, {})

    assert strcompare(reference, stream.getvalue())


@pytest.mark.serpent
def test_writeSteadyStateFile(tmp_path, beavrsMaterials):
    water = beavrsMaterials["water"]
    fuel = beavrsMaterials["fuel32"]

    comp = compBundleFromMaterials((water, fuel))
    # Only pass burnable material densities to steady state writer
    # Promote to look like multiple materials coming in
    comp = CompBundle(comp.isotopes, comp.densities[1, numpy.newaxis])

    basefile = tmp_path / "base"

    writer = hydep.serpent.SerpentWriter()
    writer.burnable = (fuel,)
    writer.base = basefile

    actual = writer.writeSteadyStateFile(
        tmp_path / "steady_state", comp, TimeStep(), 1e4,
    )

    reference = pathlib.Path(__file__).parent / "steady_state_reference"
    testfile = reference.parent / "steady_state_test"
    testfile.write_text(actual.read_text().replace(str(basefile), "BASEFILE"))

    if config["update"]:
        testfile.rename(reference)
        return

    assert filecompare(
        reference, testfile, testfile.parent / "steady_state_fail"
    )

    testfile.unlink()

    # Test EOL writer - no burnup
    eol = writer.writeSteadyStateFile(
        tmp_path / "final_steady_state", comp, TimeStep(), 1e4, final=True,
    )
    refcontent = reference.read_text().replace(" burn 1", "")
    testfile = reference.parent / "final_steady_state_test"
    testfile.write_text(eol.read_text().replace(str(basefile), "BASEFILE"))

    assert strcompare(refcontent, testfile.read_text())

    testfile.unlink()


@pytest.mark.serpent
def test_filteredMaterials(tmp_path, fakeXsDataStream):
    xsdataf = tmp_path / "fake.xsdata"
    xsdataf.write_text(fakeXsDataStream.getvalue())

    LIB = "09c"

    allIsotopes = [
        (95, 242, 0),
        (95, 242, 1),
    ]

    densities = [1e-4, 1e-5]

    explines = [
        f"95242.{LIB} {densities[0]:13.9E}",
        f"95342.{LIB} {densities[1]:13.9E}",
    ]

    writer = hydep.serpent.SerpentWriter()
    writer.updateProblemIsotopes(allIsotopes, xsdataf)

    stream = io.StringIO()

    missing = writer.writeMatIsoDef(
        stream, zip(allIsotopes, densities), LIB, threshold=0
    )
    assert missing == 0
    strcompare("\n".join(explines), stream.getvalue())

    # Check threshold
    stream.truncate(0)

    missing = writer.writeMatIsoDef(
        stream, zip(allIsotopes, densities), LIB, threshold=densities[0]
    )
    assert missing == densities[1]
    strcompare(explines[0], stream.getvalue())

    # Add an isotope that doesn't exist
    stream.truncate(0)

    BAD_ISO = (1, 200, 0)
    BAD_DENS = 1e-10
    allIsotopes.append(BAD_ISO)
    densities.append(BAD_DENS)

    p = writer.updateProblemIsotopes(allIsotopes, xsdataf)
    assert len(p.missing) == 1
    assert BAD_ISO in p.missing

    missing = writer.writeMatIsoDef(stream, zip(allIsotopes, densities), LIB)
    assert missing == BAD_DENS
    strcompare("\n".join(explines), stream.getvalue())

    xsdataf.unlink()


@pytest.mark.serpent
def test_unbounded(tmp_path, beavrsFuelPin, serpentcfg, simpleChain):
    writer = hydep.serpent.SerpentWriter()
    writer.burnable = tuple(beavrsFuelPin.findBurnableMaterials())
    writer.model = hydep.Model(beavrsFuelPin)

    assert writer.model.bounds is None
    assert writer.model.root.bounds is None

    with pytest.raises(hydep.GeometryError, match=".* unbounded.*"):
        writer.writeMainFile(tmp_path / "unbounded", serpentcfg, simpleChain)

    # Check that identical files will be written if bounds are placed
    # on root or on model
    bounds = Boundaries((-0.63, 0.63), (-0.63, 0.63), None)

    writer.model.bounds = bounds
    boundedModel = writer.writeMainFile(
        tmp_path / "bounded_model", serpentcfg, simpleChain
    )

    writer.model.bounds = None
    writer.model.root.bounds = bounds
    boundedRoot = writer.writeMainFile(
        tmp_path / "bounded_root", serpentcfg, simpleChain,
    )

    assert filecompare(boundedModel, boundedRoot)


@pytest.mark.serpent
def test_materialTemperature(writer):
    mat = hydep.Material("test", adens=1)

    # No temperature
    t = writer._getmatlib(mat)
    assert t == "06c"

    # Temperature too small -> raise a warning
    mat.temperature = 0.5 * min(writer._temps)
    with pytest.warns(hydep.DataWarning):
        t = writer._getmatlib(mat)
    assert t == "06c"

    # Use an exact temperature
    mat.temperature = 300
    t = writer._getmatlib(mat)
    assert t == "03c"

    # Use a temperature between libraries
    mat.temperature = 800
    t = writer._getmatlib(mat)
    assert t == "06c"

    # Temperature greater than max
    maxt = max(writer._temps)
    mat.temperature = maxt * 1.1
    lib = f"{maxt // 100:02}c"
    t = writer._getmatlib(mat)
    assert t == lib


@pytest.fixture
def mockSAB(mockSerpentData):
    yield mockSerpentData[Library.SAB]


@pytest.mark.serpent
def test_sab(mockSAB, writer):
    mat = hydep.Material("water", mdens=1, temperature=600)
    # Empty dictionary if no S(a, b) found
    assert writer._findSABTables([mat], mockSAB) == {}

    mat.addSAlphaBeta("HinH2O")
    actual = writer._findSABTables([mat], mockSAB)
    assert actual == {("HinH2O", "600.00"): "lwe6.12t"}

    # Off temperature material to test errors
    second = hydep.Material("other", mdens=1, temperature=608)
    second.addSAlphaBeta("HinH2O")

    with pytest.raises(hydep.DataError, match=str(second.temperature)):
        writer._findSABTables([second], mockSAB)

    with pytest.raises(hydep.DataError, match=str(second.temperature)):
        writer._findSABTables([mat, second], mockSAB)

    fake = pathlib.Path("fakesab")
    with pytest.raises(FileNotFoundError, match=".*fakesab"):
        writer._findSABTables([], fake)
