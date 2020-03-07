import os
import pathlib
from unittest.mock import Mock

import pytest
import hydep
from hydep.settings import HydepSettings
import hydep.internal
import hydep.serpent
import hydep.internal.features as hdfeat

from tests.regressions import ResultComparator, ProblemProxy


@pytest.fixture
def serpent2x2Problem(simpleChain, toy2x2lattice):
    # Include the chain so reactions are present
    model = hydep.Model(toy2x2lattice)
    model.differentiateBurnableMaterials(updateVolumes=False)

    burnable = tuple(model.root.findBurnableMaterials())
    for m in burnable:
        m.volume = 2.0  # enforce some division in volumes

    manager = Mock()
    manager.burnable = burnable
    manager.chain = simpleChain

    yield ProblemProxy(model, manager)


@pytest.fixture
def regressionSettings():
    datadir = os.environ.get("SERPENT_DATA")
    if not datadir:
        pytest.skip("Need SERPENT_DATA environment variable")
    options = {
        "hydep": {"archive on success": True, "boundary conditions": "reflective"},
        "hydep.serpent": {
            "seed": 12345678910,
            "particles": 100,
            "generations per batch": 2,
            "active": 10,
            "inactive": 2,
            "executable": "sss2",
            "acelib": "sss_endfb7u.xsdata",
            "declib": "sss_endfb7.dec",
            "nfylib": "sss_endfb7.nfy",
            "datadir": datadir,
        },
    }
    settings = HydepSettings()
    settings.updateAll(options)
    return settings


@pytest.fixture
def serpentSolver(runInTempDir, regressionSettings, serpent2x2Problem):
    regressionSettings.rundir = runInTempDir

    serpentDir = runInTempDir / "serpent"
    assert not serpentDir.exists()

    # Set hooks for slightly realistic problem
    XS_KEYS = {"abs", "fiss"}
    hooks = hdfeat.FeatureCollection(
        {hdfeat.HOMOG_LOCAL, hdfeat.FISSION_MATRIX, hdfeat.FISSION_YIELDS}, XS_KEYS
    )

    solver = hydep.serpent.SerpentSolver()

    solver.setHooks(hooks)

    assert solver.hooks == hooks

    solver.beforeMain(
        serpent2x2Problem.model, serpent2x2Problem.manager, regressionSettings
    )
    assert serpentDir.is_dir()

    timeStep = hydep.internal.TimeStep(0, 0, 0, 0)

    concentrations = hydep.internal.compBundleFromMaterials(
        serpent2x2Problem.manager.burnable
    )

    # Set a realistic power for this time step

    POWER = 6e6

    solver.bosUpdate(concentrations, timeStep, POWER)
    yield solver
    solver.finalize(True)

@pytest.mark.serpent
def test_serpentSolver(serpentSolver):

    serpentSolver.execute()

    res = serpentSolver.processResults()

    tester = ResultComparator(pathlib.Path(__file__).parent)
    tester.main(res)
