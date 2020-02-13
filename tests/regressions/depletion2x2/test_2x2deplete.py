import math
import pathlib
from collections import namedtuple

import numpy
import pytest
import hydep
import hydep.internal

from tests.regressions import ProblemProxy
from . import DepletionComparator


DepBundle = namedtuple("DepBundle", "manager reactionRates fissionYields")
THERMAL_ENERGY = 0.0253


def buildFissionYields(chain, ene=THERMAL_ENERGY, fallbackIndex=0):
    allYields = {}
    for isotope in chain:
        if isotope.fissionYields is None:
            continue
        isoYields = isotope.fissionYields.get(ene)
        if isoYields is None:
            isoYields = isotope.fissionYields.at(fallbackIndex)
        allYields[isotope.zai] = isoYields
    return allYields


@pytest.fixture
def depletionModel():
    # Roughly one-fourth volume of fuel in PWR
    VOLUME = math.pi * 0.49 ** 2 * 360 * 264 / 4
    fuel1 = hydep.BurnableMaterial(
        "partially burned PWR fuel", adens=2.6858e-2, temperature=900, volume=VOLUME
    )

    ISO_THRESH = 1e-10
    CONC_FILE = pathlib.Path(__file__).parent / "partial_burned.dat"
    with CONC_FILE.open("r") as stream:
        for line in stream:
            za, adens = line.split()
            adens = float(adens)
            if adens < ISO_THRESH:
                continue
            z, a = divmod(int(za), 1000)
            fuel1[(z, a, 0)] = adens

    water = hydep.Material("water", mdens=0.7, H1=2, O16=1)
    clad = hydep.Material("clad", mdens=6.6, Zr96=1)

    fuel2 = fuel1.copy()
    fuel3 = fuel1.copy()
    fuel4 = hydep.BurnableMaterial(
        "other fuel",
        adens=fuel1.adens,
        temperature=fuel1.temperature,
        volume=VOLUME,
        O16=4.643355421e-4,
        U234=4.742255584e-9,
        U235=7.403701961e-4,
        U236=1.090905903e-5,
        U238=2.550365361e-2,
    )

    pin1 = hydep.Pin([0.39, 0.42], [fuel1, clad], outer=water)
    pin2 = hydep.Pin([0.39, 0.42], [fuel2, clad], outer=water)
    pin3 = hydep.Pin([0.39, 0.42], [fuel3, clad], outer=water)
    pin4 = hydep.Pin([0.39, 0.42], [fuel4, clad], outer=water)

    PITCH = 1.2

    cart = hydep.CartesianLattice(
        nx=2, ny=2, pitch=1.2, array=[[pin1, pin2], [pin3, pin4]], outer=water
    )

    assembly = hydep.LatticeStack(1, [0, 360], [cart])
    assembly.bounds = ((-PITCH, PITCH), (-PITCH, PITCH), (0, 360))

    # TODO Retrieve boundaries from model if given
    model = hydep.Model(assembly)
    model.bounds = assembly.bounds

    yield ProblemProxy(model, tuple(model.root.findBurnableMaterials()))


@pytest.fixture
def depletionHarness(endfChain, depletionModel):

    N_BURNABLE = len(depletionModel.burnable)
    N_GROUPS = 1

    # Get microscopic cross sections, flux

    datadir = pathlib.Path(__file__).parent
    fluxes = numpy.loadtxt(datadir / "fluxes.dat").reshape(N_BURNABLE, N_GROUPS)

    fissionYields = hydep.internal.FakeSequence(
        buildFissionYields(endfChain, ene=THERMAL_ENERGY, fallbackIndex=0), N_BURNABLE
    )

    microxs = []

    for ix in range(N_BURNABLE):
        mxsfile = datadir / "mxs{}.dat".format(ix + 1)
        assert mxsfile.is_file()
        flux = fluxes[ix] / depletionModel.burnable[ix].volume
        mxsdata = numpy.loadtxt(mxsfile)
        zai = mxsdata[:, 0].astype(int)
        rxns = mxsdata[:, 1].astype(int)
        mxs = mxsdata[:, 2:2 + 1 + N_GROUPS]

        microxs.append(
            hydep.internal.MicroXsVector.fromLongFormVectors(
                zai, rxns, mxs * flux, assumeSorted=True
            )
        )

    timestep = 50
    power = 6e6
    divisions = 1
    manager = hydep.Manager(endfChain, [timestep], [power], divisions)
    manager.beforeMain(depletionModel.model)

    yield DepBundle(manager, microxs, fissionYields)


@pytest.mark.flaky
def test_2x2deplete(depletionHarness):
    manager = depletionHarness.manager
    out = manager.deplete(
        manager.timesteps[0],
        depletionHarness.reactionRates,
        depletionHarness.fissionYields,
    )

    compare = DepletionComparator(pathlib.Path(__file__).parent)
    compare.main(out)
