import os
import pathlib
import tempfile

import pytest
import hydep

from tests.regressions import ProblemProxy


@pytest.fixture
def toy2x2Problem(simpleChain, toy2x2lattice):
    # Include the chain so reactions are present
    model = hydep.Model(toy2x2lattice)
    model.differentiateBurnableMaterials(updateVolumes=False)

    burnable = tuple(model.root.findBurnableMaterials())
    for m in burnable:
        m.volume = 1.0

    yield ProblemProxy(model, burnable)


@pytest.fixture
def runInTempDir():
    """Inspired by similar openmc regression fixture"""
    pwd = pathlib.Path.cwd()

    with tempfile.TemporaryDirectory() as tdir:
        os.chdir(tdir)
        yield pathlib.Path(tdir)
        os.chdir(pwd)
