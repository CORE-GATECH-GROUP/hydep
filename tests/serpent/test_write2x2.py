"""Test the ability to write an assymetric 2x2 bundle"""

import pathlib
import io

import hydep
import hydep.serpent
from tests.serpent import filecompare


def test_write2x2(serpentcfg, beavrsFuelPin, beavrsControlPin, beavrsGuideTube):
    template = [[0, 0], [1, 2]]
    fill = {0: beavrsFuelPin, 1: beavrsGuideTube, 2: beavrsControlPin}
    asymmetric2x2 = hydep.CartesianLattice.fromMask(1.3, template, fill)

    model = hydep.Model(asymmetric2x2)
    burnable = tuple(model.findBurnableMaterials())
    assert len(burnable) == 2

    writer = hydep.serpent.SerpentWriter()
    writer.burnable = burnable
    writer.model = model
    writer.configure(serpentcfg, 3)

    assert writer.options["bc"] == [2, 2, 2]

    rundir = pathlib.Path(__file__).parent
    output = rundir / "2x2.txt"
    reference = rundir / "2x2_ref.txt"
    failfile = rundir / "2x2_fail.txt"

    writer.writeBaseFile(output)

    assert filecompare(reference, output, failfile)

    output.unlink()
