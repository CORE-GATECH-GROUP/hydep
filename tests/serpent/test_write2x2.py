"""Test the ability to write an assymetric 2x2 bundle"""

import pathlib

import pytest
import hydep
import hydep.serpent

from tests import filecompare


@pytest.mark.serpent
def test_write2x2(serpentcfg, write2x2Model):
    burnable = tuple(write2x2Model.root.findBurnableMaterials())
    assert len(burnable) == 2

    writer = hydep.serpent.SerpentWriter()
    writer.burnable = burnable
    writer.model = write2x2Model
    writer.configure(serpentcfg, 3)

    assert writer.options["bc"] == [2, 2, 2]

    rundir = pathlib.Path(__file__).parent
    output = rundir / "2x2.txt"
    reference = rundir / "2x2_ref.txt"
    failfile = rundir / "2x2_fail.txt"

    writer.writeBaseFile(output)

    written = output.read_text()
    for key in {"acelib", "nfylib", "declib"}:
        written = written.replace(serpentcfg[key], key.upper())
    failfile.write_text(written)

    assert filecompare(reference, failfile, failfile)

    output.unlink()
    failfile.unlink()
