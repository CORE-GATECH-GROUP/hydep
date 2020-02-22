import io
import pathlib

import numpy
import pytest
import hydep
from hydep.internal import CompBundle, TimeStep, compBundleFromMaterials
import hydep.serpent

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

    reference = pathlib.Path(__file__).parent / "steady_state_refefence"
    testfile = reference.parent / "steady_state_test"
    testfile.write_text(actual.read_text().replace(str(basefile), "BASEFILE"))

    if config["update"]:
        testfile.rename(reference)
        return

    assert filecompare(reference, testfile, testfile.parent / "steady_state_fail")

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
