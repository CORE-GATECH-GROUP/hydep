import pathlib

import pytest
from hydep.constants import SECONDS_PER_DAY
from hydep.sfv import SfvSolver
from hydep.internal import TimeStep

from . import SfvComparator


@pytest.mark.sfv
def test_sfvsimple(simpleSfvProblem, sfvMacroData, sfvMicroXS, sfvNewComps):

    solver = SfvSolver(numModes=len(simpleSfvProblem.burnable), fittingOrder=0)
    solver.beforeMain(simpleSfvProblem.model, simpleSfvProblem.burnable)

    # Emulate BOS result
    bos = sfvMacroData.toTransportResult()
    timestep = TimeStep(0, 0, 0, 0)
    POWER = 1.16135E4

    solver.processBOS(bos, timestep, POWER)

    # Test beginning of step data
    assert (solver.macroAbs0 == sfvMacroData.siga0).all()
    assert (solver.macroNsf0 == sfvMacroData.nsf0).all()
    # How to test normalized phi 0?

    timestep += 50 * SECONDS_PER_DAY

    solver.substepUpdate(timestep, sfvNewComps, sfvMicroXS)
    solver.execute()
    flux = solver.processResults().flux[:, 0]

    comparator = SfvComparator(pathlib.Path(__file__).parent, sfvMacroData)

    comparator.main(
        phi1=flux,
        siga1=solver.macroAbs1,
        sigf1=solver.macroFis1,
        nubar1=solver.extrapolatedNubar,
        kappaSigf1=solver.kappaSigf1,
    )
