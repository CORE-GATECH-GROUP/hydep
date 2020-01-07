import configparser
import pathlib

import pytest
from hydep.internal import configmethod


class ConfigureTester:
    data = {
        "hydep": {"archive on success": "true"},
        "hydep.serpent": {"executable": "sss2"},
    }

    def __init__(self):
        self.configfile = pathlib.Path(__file__).parent / "configtest_ref.cfg"

    def setup(self):
        parser = configparser.ConfigParser()
        parser.read_dict(self.data)
        with self.configfile.open("w") as stream:
            parser.write(stream)

    def teardown(self):
        self.configfile.unlink()

    @configmethod
    def launchTest(self, cfg):
        assert isinstance(cfg, configparser.ConfigParser)
        assert set(cfg.sections()) == set(self.data)
        for section, sub in self.data.items():
            assert cfg.has_section(section)
            assert set(cfg.options(section)) == set(sub)
            for option, value in sub.items():
                assert cfg.get(section, option) == value
        return cfg

    def mapping(self):
        return self.data

    def filename(self):
        return str(self.configfile)

    def path(self):
        return self.configfile

    def parser(self):
        parser = configparser.ConfigParser()
        parser.read(str(self.configfile))
        return parser


@pytest.fixture(scope="module")
def configHarness():
    harness = ConfigureTester()
    harness.setup()
    yield harness
    harness.teardown()


@pytest.mark.parametrize("how", ("mapping", "filename", "path", "parser"))
def test_configmethod(configHarness, how):
    userOptions = getattr(configHarness, how)()
    configHarness.launchTest(userOptions)
