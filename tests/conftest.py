"""Configuration options and fixtures for testing"""
import pathlib

import pytest
import hydep
from hydep import DepletionChain

from tests.regressions import config as regressionCfg


def pytest_addoption(parser):
    parser.addoption("--update", action="store_true")


def pytest_configure(config):
    upd = config.getoption("update")
    if upd is not None:
        regressionCfg["update"] = upd


@pytest.fixture(scope="session")
def simpleChain():
    chainfile = pathlib.Path(__file__).parent / "simple_chain.xml"
    yield DepletionChain.fromXml(str(chainfile))


@pytest.fixture
def toy2x2lattice():
    """A simple 2x2 model with 4 UO2 fuel pins"""
    fuel = hydep.BurnableMaterial("fuel", mdens=10.4, U235=8e-4, U238=2e-2, O16=4e-4)
    clad = hydep.Material("clad", mdens=6, Zr91=4e-2)
    water = hydep.Material("water", mdens=1, H1=5e-2, O16=2e-2)

    pin = hydep.Pin([0.42, 0.45], [fuel, clad], outer=water)

    return hydep.CartesianLattice(2, 2, 1.23, [[pin, pin], [pin, pin]])
