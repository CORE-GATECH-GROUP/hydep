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
def serpentSolver(tmpdir):
    options = {
        "hydep": {"archive on success": True},
        "hydep.serpent": {
            "random seed": 12345678910,
            "boundary conditions": "reflective",
            "particles": 100,
            "generations per batch": 2,
            "active": 5,
            "skipped": 2,
            "executable": "sss2",
        },
    }

    solver = hydep.serpent.SerpentSolver()
    solver.configure(options)

    with tmpdir.as_cwd():
        tmpdir.mkdir("serpent")
        yield solver
        solver.finalize(True)

@pytest.fixture
def runInTempDir():
    """Inspired by similar openmc regression fixture"""
    pwd = pathlib.Path.cwd()

    with tempfile.TemporaryDirectory() as tdir:
        os.chdir(tdir)
        yield pathlib.Path(tdir)
        os.chdir(pwd)
