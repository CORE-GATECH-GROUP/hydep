"""Configuration options for pytest"""

from tests.regressions import config as regressionCfg

def pytest_addoption(parser):
    parser.addoption("--update", action="store_true")


def pytest_configure(config):
    upd = config.getoption("update")
    if upd is not None:
        regressionCfg["update"] = upd
