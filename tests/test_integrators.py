"""Test the integrators using a simple system

Two isotope system, with isotope A initially present and B absent.
The transmutation chain is A (n,gamma) -> B (n,gamma) -> removed,
and contained in a companion xml file. The isotopes are named U235
and Xe135 because the chain and framework expects real isotope names.
For this problem then, A will be represented by U235 and B by Xe135.

The system will sovle across a single depletion step of five seconds,
and have fixed microscopic cross sections RA and RB. The "transport
solution" is obtained as a function
``phi(NA, NB) = cos(pi * NA / 4) + sin(pi * NB)``. This allows a
"realistic"ly structured depletion matrix to be built using two reaction
rates and all the under-lying depletion data. The system of ODEs is then::

    A' = -RA * phi(NA, NB) * NA
    B' =  RA * phi(NA, NB) * NA - RB * phi(NA, NB) * NB

for ``t in [0, 5]`` and initial conditions [A, B](0) = [1, 0]
The final compositions as computed by scipy.integrate.solve_ivp using
an adaptive RK4/5 are ``[A, B](5) = [0.0105478107949492 0.4873257599455337]``
"""

import collections
import pathlib
import math

import numpy
import pytest
import hydep
from hydep.constants import SECONDS_PER_DAY
import hydep.internal
import hydep.internal.features as hdfeat


def buildDepletionChain():
    chainfile = pathlib.Path(__file__).parent / "analytic_chain.xml"
    return hydep.DepletionChain.fromXml(chainfile)


class AnalyticStore(hydep.lib.BaseStore):
    """Class that holds densities for analytic time integration"""
    def __init__(self):
        self.densities = []

    @staticmethod
    def beforeMain(*args, **kwargs):
        pass

    @staticmethod
    def postTransport(*args, **kwargs):
        pass

    def writeCompositions(self, _timestep, compBundle):
        self.densities.append(compBundle.densities[0])


def transportSolution(compositions):
    """Mimic a transport solution by varying the flux using compositions"""

    index = hydep.internal.XsIndex([541350, 922350], [102, 102], [0, 1, 2])
    mxs = numpy.array([[0.1, 0.5]])
    flux = (
        math.cos(compositions[1] * math.pi * 0.25)
        + math.sin(compositions[0] * math.pi)
    )
    return hydep.internal.TransportResult(
        flux=[[flux]],
        keff=[0.0, 0.0],
        fissionYields=[{}],
        microXS=hydep.internal.MaterialDataArray(index, mxs),
    )


class AnalyticHFSolver(hydep.lib.HighFidelitySolver):
    """Concrete interface for high fidelity solver"""
    features = hdfeat.FeatureCollection(
        {hdfeat.MICRO_REACTION_XS, hdfeat.FISSION_YIELDS},
    )

    @staticmethod
    def bosSolve(compositions, timestep, _power):
        return transportSolution(compositions.densities[0])

    @staticmethod
    def setHooks(needs):
        pass


class AnalyticROSolver(hydep.lib.ReducedOrderSolver):
    """Interface for reduced-order solver

    Provides the exact flux given the compositions in the
    same manner as the high-fidelity solver.
    """
    needs = hdfeat.FeatureCollection()

    @staticmethod
    def substepSolve(timestep, compositions, _mxs):
        return transportSolution(compositions.densities[0])


# Xe135 and U235 are the EOS concentration (atoms/b-cm) after a 5 second
# depletion event. These are the expected values given each time integration
# scheme, not strictly the "true" solution to the ODE
SolverBundle = collections.namedtuple("SolverBundle", "solver xe135 u235")


SCHEMES = {
    "predictor": SolverBundle(
        hydep.PredictorIntegrator, 0.6643434074084894, 0.170713775399768
    ),
    "celi": SolverBundle(hydep.CELIIntegrator, 0.6073748087895677, 0.0403675404653474),
    "rk4": SolverBundle(hydep.RK4Integrator, 0.5111748628262343, 0.0134443028193424),
}


@pytest.fixture
def model():
    mat = hydep.BurnableMaterial("analytic", adens=1.0, volume=1.0)
    mat["U235"] = 1.0

    return hydep.Model(hydep.InfiniteMaterial(mat))


@pytest.fixture
def manager(clearIsotopes):
    dep = hydep.Manager(buildDepletionChain(), [5 / SECONDS_PER_DAY], [1.0], [1])

    return dep


@pytest.mark.parametrize("scheme", SCHEMES)
def test_integrator(recwarn, model, manager, scheme):
    store = AnalyticStore()
    klass, eosXe, eosU = SCHEMES[scheme]
    solver = klass(model, AnalyticHFSolver(), AnalyticROSolver(), manager, store=store)

    solver.integrate()
    assert store.densities[-1] == pytest.approx([eosXe, eosU])


@pytest.mark.parametrize("scheme", ("celi", "rk4"))
@pytest.mark.parametrize("brave", (True, False))
def test_brave_integrators(recwarn, model, manager, scheme, brave):
    store = AnalyticStore()
    klass, eosXe, eosU = SCHEMES[scheme]
    solver = klass(
        model,
        AnalyticHFSolver(),
        AnalyticROSolver(),
        manager,
        store=store,
        brave=brave
    )

    solver.integrate()

    if not brave:
        w = recwarn.pop(hydep.ExperimentalIntegratorWarning)
        assert klass.__name__ in str(w.message)
    assert len(recwarn) == 0
    assert store.densities[-1] == pytest.approx([eosXe, eosU])
