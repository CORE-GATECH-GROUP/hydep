import os
import math

import pytest
import hydep
import hydep.simplerom

from . import HdfResultCompare

hdserpent = pytest.importorskip("hydep.serpent")
hdfstore = pytest.importorskip("hydep.hdfstore")


PATCHED_EXE = "sss2-extdep"


@pytest.fixture
def pincell():
    fuel = hydep.BurnableMaterial("fuel", mdens=10.4, volume=math.pi * 0.39 * 0.39)
    fuel["O16"] = 4.6391716e-2
    fuel["U234"] = 9.3422610e-6
    fuel["U235"] = 1.0452130e-3
    fuel["U236"] = 4.7875776e-6
    fuel["U238"] = 2.2145310e-2

    water = hydep.Material("water", mdens=1, temperature=600)
    water["H1"] = 5.01543e-2
    water["O16"] = 2.50771e-2
    water.addSAlphaBeta("HinH2O")

    clad = hydep.Material("clad", mdens=6.6, temperature=600)
    clad["Fe56"] = 1.24985e-2
    clad["H1"] = 3.34334e-2
    clad["O16"] = 1.66170e-2

    pin = hydep.Pin([0.39, 0.402], [fuel, clad], outer=water)

    HPITCH = 0.63
    model = hydep.Model(pin)
    model.bounds = pin.bounds = ((-HPITCH, HPITCH), (-HPITCH, HPITCH), None)
    return model


@pytest.mark.serpent
def test_external(runInTempDir, endfChain, pincell):
    datadir = os.environ.get("SERPENT_DATA")
    if not datadir:
        pytest.skip("Need SERPENT_DATA environment variable")
    serpent = hdserpent.CoupledSerpentSolver()
    simplerom = hydep.simplerom.SimpleROSolver()

    manager = hydep.Manager(endfChain, [50], 1e3, substepDivision=1)
    problem = hydep.Problem(pincell, serpent, simplerom, manager)

    problem.configure(
        {
            "hydep": {
                "boundary conditions": "reflective",
                "basedir": runInTempDir,
                "rundir": None,
                "use temp dir": True,
            },
            "hydep.serpent": {
                "seed": 123456789,
                "executable": PATCHED_EXE,
                "particles": 100,
                "generations per batch": 10,
                "active": 2,
                "inactive": 5,
                "acelib": "sss_endfb7u.xsdata",
                "declib": "sss_endfb7.dec",
                "nfylib": "sss_endfb7.nfy",
                "datadir": datadir,
            },
        }
    )

    tester = HdfResultCompare()
    tester.main(problem)
