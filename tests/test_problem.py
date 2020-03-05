"""Basic unit testing for problem"""

import configparser

import pytest
import hydep
from hydep.lib import HighFidelitySolver, ReducedOrderSolver
from hydep.settings import SubSetting, HydepSettings


class MockHFSettings(SubSetting, sectionName="mockHF"):
    def __init__(self):
        self.foo = None

    def update(self, options):
        foo = options.pop("foo", None)
        if foo is not None:
            self.foo = foo


class MockHFSolver(HighFidelitySolver):
    features = None

    @staticmethod
    def execute():
        return

    @staticmethod
    def processResults():
        return

    @staticmethod
    def bosUpdate(*args, **kwargs):
        pass

    @staticmethod
    def setHooks(*args, **kwargs):
        pass

    @staticmethod
    def checkCompatibility(*args, **kwargs):
        return True


class MockROMSolver(ReducedOrderSolver):
    needs = None

    @staticmethod
    def execute():
        return

    @staticmethod
    def processResults():
        return

    @staticmethod
    def substepUpdate(*args, **kwargs):
        pass


class MockRomSettings(SubSetting, sectionName="mockROM"):
    def __init__(self):
        self.bar = None

    def update(self, options):
        bar = options.pop("bar", None)
        if bar is not None:
            self.bar = bar


@pytest.fixture
def mockProblem():
    fuel = hydep.BurnableMaterial("fuel", mdens=10.4, volume=1.0, U235=1e-5)
    univ = hydep.InfiniteMaterial(fuel)
    model = hydep.Model(univ)

    # manager
    chain = hydep.DepletionChain(fuel.keys())
    manager = hydep.Manager(chain, [10], 1e6, 1)

    return hydep.Problem(model, MockHFSolver(), MockROMSolver(), manager)


@pytest.fixture
def settings():
    return {
        "hydep": {
            "archive on success": True,
            "depletion solver": "cram48",
            "boundary conditions": ("reflective", "reflective", "vacuum"),
        },
        "hydep.mockHF": {"foo": "foo",},
        "hydep.mockROM": {"bar": "bar",},
    }


def _testSettings(settings):
    assert isinstance(settings, HydepSettings)
    assert settings.archiveOnSuccess
    assert settings.depletionSolver == "cram48"
    assert tuple(settings.boundaryConditions) == ("reflective", "reflective", "vacuum")
    assert settings.mockHF.foo == "foo"
    assert settings.mockROM.bar == "bar"


def test_dictconfig(mockProblem, settings):
    mockProblem.configure(settings)
    _testSettings(mockProblem.settings)


@pytest.fixture
def fileConfig(tmp_path, settings):
    fp = tmp_path / "mock.cfg"
    cfg = configparser.ConfigParser()
    bc = settings["hydep"]["boundary conditions"]
    settings["hydep"]["boundary conditions"] = " ".join(bc)
    cfg.read_dict(settings)
    with fp.open("w") as stream:
        cfg.write(stream)
    yield fp
    fp.unlink()


def test_fileConfig(mockProblem, fileConfig):
    mockProblem.configure(fileConfig)
    _testSettings(mockProblem.settings)
