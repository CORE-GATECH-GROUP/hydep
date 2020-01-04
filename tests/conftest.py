"""Configuration options for pytest"""
import pathlib

import pytest
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
