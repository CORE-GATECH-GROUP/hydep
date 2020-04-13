import pathlib
import math
from unittest.mock import Mock

import numpy
import pytest
import hydep
from hydep.constants import CM2_PER_BARN
from hydep.internal import MicroXsVector, CompBundle, getIsotope

from tests.regressions import ProblemProxy
from . import SfvDataHarness



@pytest.fixture
def simpleSfvProblem(endfChain):
    PIN_RAD = 0.39
    PIN_HEIGHT = 10
    PITCH = 1.2

    volume = math.pi * PIN_RAD ** 2 * PIN_HEIGHT

    fuel1 = hydep.BurnableMaterial(
        "partial burned PWR fuel", adens=2.6856e-2, temperature=900, volume=volume
    )

    fuel1[80160] = 4.643e-04
    fuel1[922340] = 4.742e-9
    fuel1[922350] = 7.404e-4
    fuel1[922360] = 1.091e-5
    fuel1[922380] = 2.550e-5
    fuel1[942390] = 2.569e-05
    fuel1[942400] = 1.099e-06
    fuel1[942410] = 1.225e-07
    fuel1[942420] = 1.866e-09

    fuel2 = fuel1.copy()
    fuel3 = fuel1.copy()

    fuel4 = hydep.BurnableMaterial(
        "other fuel",
        adens=fuel1.adens,
        temperature=fuel1.temperature,
        volume=volume,
        O16=4.6433e-4,
        U234=4.742e-9,
        U235=7.403e-4,
        U236=1.091e-5,
        U238=2.55e-2,
    )

    water = hydep.Material("water", mdens=0.7, H1=2, O16=1)
    clad = hydep.Material("clad", mdens=6.6, Zr96=1)

    pin1 = hydep.Pin([PIN_RAD, 0.42], [fuel1, clad], outer=water)
    pin2 = hydep.Pin([PIN_RAD, 0.42], [fuel2, clad], outer=water)
    pin3 = hydep.Pin([PIN_RAD, 0.42], [fuel3, clad], outer=water)
    pin4 = hydep.Pin([PIN_RAD, 0.42], [fuel4, clad], outer=water)

    cart = hydep.CartesianLattice(
        nx=2, ny=2, pitch=1.2, array=[[pin1, pin2], [pin3, pin4]], outer=water
    )

    assembly = hydep.LatticeStack(2, [0, PIN_HEIGHT, 2 * PIN_HEIGHT], [cart, cart])
    assembly.bounds = ((-PITCH, PITCH), (-PITCH, PITCH), (0, 2 * PIN_HEIGHT))

    model = hydep.Model(assembly)
    model.bounds = assembly.bounds
    model.root.differentiateBurnableMaterials()

    problem = Mock()
    problem.model = model
    problem.dep.burnable = tuple(model.root.findBurnableMaterials())
    problem.dep.chain = endfChain

    settings = hydep.settings.Settings(fittingOrder=1, numFittingPoints=2)
    settings.sfv.densityCutoff = 0.0
    settings.sfv.modes = len(problem.dep.burnable)
    problem.settings = settings

    yield problem


@pytest.fixture
def sfvMacroData():
    datadir = pathlib.Path(__file__).parent
    macrofile = datadir / "macroxs.dat"
    fmtxfile = datadir / "fmtx.dat"
    return SfvDataHarness.fromDataFiles(macrofile, fmtxfile)


@pytest.fixture
def sfvMicroXS(simpleSfvProblem):
    mxs = []
    datadir = pathlib.Path(__file__).parent
    for m in simpleSfvProblem.dep.burnable:
        mxsf = datadir / f"mxs{m.id}.dat"
        data = numpy.loadtxt(mxsf)
        mxs.append(
            MicroXsVector.fromLongFormVectors(
                data[:, 0].astype(int),
                data[:, 1].astype(int),
                data[:, 2] * CM2_PER_BARN,
                assumeSorted=False,
            )
        )
    return mxs


@pytest.fixture
def sfvNewComps():
    datafile = pathlib.Path(__file__).parent / "compositions.dat"

    data = numpy.asfortranarray(numpy.loadtxt(datafile))
    isotopes = tuple(getIsotope(zai=z) for z in data[:, 0].astype(int))
    densities = [data[:, ix] for ix in range(1, data.shape[1])]

    return CompBundle(isotopes, densities)
