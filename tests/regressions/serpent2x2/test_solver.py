import pathlib

import pytest
import hydep
import hydep.internal
import hydep.serpent
import hydep.internal.features as hdfeat

from tests.regressions import ResultComparator


@pytest.mark.serpent
def test_serpentSolver(serpentSolver, toy2x2Problem):
    model = toy2x2Problem.model
    burnable = toy2x2Problem.burnable

    # Set hooks for slightly realistic problem
    XS_KEYS = {"abs", "fiss"}
    hooks = hdfeat.FeatureCollection(
        {hdfeat.HOMOG_LOCAL, hdfeat.FISSION_MATRIX}, XS_KEYS
    )

    serpentSolver.setHooks(hooks)

    assert serpentSolver.hooks == hooks

    serpentSolver.beforeMain(model, burnable)

    timeStep = hydep.internal.TimeStep(0, 0, 0, 0)

    # Concentrations for the current step
    # Not sure how these will look in the future
    # Set to None to catch errors down the road

    DUMMY_CONCS = None

    # Set a realistic power for this time step

    POWER = 6e6

    serpentSolver.bosUpdate(DUMMY_CONCS, timeStep, POWER)

    serpentSolver.execute()

    res = serpentSolver.processResults()

    tester = ResultComparator(pathlib.Path(__file__).parent)
    tester.main(res)
