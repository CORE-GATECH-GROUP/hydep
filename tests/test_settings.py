import random
import string
import pathlib
from unittest.mock import patch

import pytest
from hydep.settings import HydepSettings, SubSetting, asBool


def test_settings():
    ARCHIVE = "archive on success"
    DEP_SOLVER = "depletion solver"
    BC = "boundary conditions"
    FIT_ORDER = "fitting order"
    N_FIT_POINTS = "fitting points"
    UNBOUND_FIT = "unbounded fitting"

    h = HydepSettings()
    h.update(
        {
            ARCHIVE: "true",
            DEP_SOLVER: "cram48",
            BC: "reflective reflective vacuum",
            FIT_ORDER: 2,
            N_FIT_POINTS: 4,
            UNBOUND_FIT: False,
        }
    )

    assert h.archiveOnSuccess
    assert h.depletionSolver == "cram48"
    assert h.boundaryConditions == ("reflective", "reflective", "vacuum")
    assert h.fittingOrder == 2
    assert h.numFittingPoints == 4
    assert not h.unboundedFitting

    h.validate()

    h.update(
        {
            ARCHIVE: "0",
            BC: "vacuum",
            FIT_ORDER: "1",
            N_FIT_POINTS: "none",
            UNBOUND_FIT: "y",
        }
    )
    assert not h.archiveOnSuccess
    assert h.boundaryConditions == ("vacuum", ) * 3
    assert h.fittingOrder == 1
    assert h.numFittingPoints is None
    assert h.unboundedFitting

    with pytest.raises(TypeError, match=".*archive.*bool"):
        h.update({ARCHIVE: "positive"})
    assert not h.archiveOnSuccess

    with pytest.raises(ValueError, match=".*[B|b]oundary"):
        HydepSettings().update({BC: ["reflective", "very strange", "vacuum"]})
    assert h.boundaryConditions == ("vacuum", ) * 3

    with pytest.raises(TypeError):
        h.archiveOnSuccess = 1

    with pytest.raises(TypeError):
        h.unboundedFitting = 1

    fresh = HydepSettings(
        archiveOnSuccess=True,
        depletionSolver="testSolver",
        boundaryConditions="reflective",
    )
    assert fresh.boundaryConditions == ("reflective", ) * 3
    assert fresh.archiveOnSuccess
    assert fresh.depletionSolver == "testSolver"


def test_validate():
    with pytest.raises(ValueError):
        HydepSettings(fittingOrder=2, numFittingPoints=1).validate()

    with pytest.raises(ValueError):
        HydepSettings(numFittingPoints=1, unboundedFitting=True).validate()

    # Only enforce if number of previous points is definitely given
    HydepSettings(numFittingPoints=None, unboundedFitting=True).validate()


def test_subsettings():
    randomSection = "".join(random.sample(string.ascii_letters, 10))
    settings = HydepSettings()
    assert not hasattr(settings, "test")
    assert not hasattr(settings, randomSection)

    class IncompleteSetting(SubSetting, sectionName="incomplete"):
        pass

    with pytest.raises(TypeError, match=".*abstract methods"):
        IncompleteSetting()

    class MySubSettings(SubSetting, sectionName="test"):
        def __init__(self):
            self.truth = True

        def update(self, options):
            v = options.get("truth", None)
            if v is not None:
                self.truth = asBool("truth", v)

    t = settings.test
    assert isinstance(t, MySubSettings)
    assert not hasattr(settings, randomSection)
    assert t.truth

    settings.updateAll({"hydep.test": {"truth": "0"}})
    assert not t.truth

    fresh = HydepSettings()
    fresh.updateAll(
        {"hydep": {"depletion solver": "cram48"}, "hydep.test": {"truth": "n"}}
    )
    assert fresh.depletionSolver == "cram48"
    assert not fresh.test.truth

    with pytest.raises(ValueError, match=f".*{randomSection}"):
        fresh.updateAll({f"hydep.{randomSection}": {"key": "value"}})

    with pytest.raises(ValueError, match=".*test"):

        class DuplicateSetting(SubSetting, sectionName="test"):
            pass


@pytest.mark.parametrize(
    "name", ("0hello", "hello world", "hydep.serpent", "mock-test", "w!ld3xample")
)
def test_badSubsectionNames(name):
    with pytest.raises(ValueError, match=f".*{name}"):

        class Failure(SubSetting, sectionName=name):
            pass


def test_directories(tmpdir):
    PWD = pathlib.Path.cwd()
    FAKE_DIR = pathlib.Path(tmpdir)

    settings = HydepSettings(basedir=FAKE_DIR, rundir=None)

    assert settings.basedir.is_absolute()
    assert settings.basedir == FAKE_DIR
    assert settings.rundir is None
    assert not settings.useTempDir

    # Resolution relative to current working directory
    settings.rundir = FAKE_DIR.name
    assert settings.rundir.is_absolute()
    assert settings.rundir == PWD / FAKE_DIR.name

    # String contains an absolute path
    settings.basedir = str(FAKE_DIR)
    assert settings.basedir.is_absolute()
    assert settings.basedir == FAKE_DIR

    # Check defaulting to CWD
    fresh = HydepSettings(basedir=None)
    assert fresh.basedir.is_absolute()
    assert fresh.basedir == pathlib.Path.cwd()

    with pytest.raises(TypeError):
        fresh.basedir = None
    assert fresh.basedir == pathlib.Path.cwd()

    fresh.update(
        {
            "basedir": FAKE_DIR.name,
            "rundir": "nONe",
            "use temp dir": "true",
        }
    )

    assert fresh.basedir == PWD / FAKE_DIR.name
    assert fresh.basedir.is_absolute()
    assert fresh.rundir is None
    assert fresh.useTempDir

    fresh.update({"rundir": FAKE_DIR.name, "use temp dir": "falSe"})

    assert fresh.rundir == PWD / FAKE_DIR.name
    assert fresh.rundir.is_absolute()
    assert not fresh.useTempDir

    with pytest.raises(TypeError):
        fresh.update({"basedir": "none"})


@pytest.fixture
def serpentdata(tmpdir):
    datadir = pathlib.Path(tmpdir / "serpentdata")
    datadir.mkdir()

    # Must match up with example config file
    files = {"sss_endfb7u.xsdata", "sss_endfb71.dec", "sss_endfb71.nfy"}

    for name in files:
        (datadir / name).touch()

    with patch.dict(
        "os.environ", {"SERPENT_DATA": str(datadir), "OMP_NUM_THREADS": ""}
    ):
        yield datadir

    for name in files:
        (datadir / name).unlink()


def test_exampleConfig(serpentdata):
    cfg = pathlib.Path(__file__).parents[1] / "hydep.cfg.example"
    assert cfg.is_file(), cfg

    settings = HydepSettings.fromFile(cfg)

    assert settings.boundaryConditions == ("reflective", "vacuum", "reflective")
    assert settings.basedir == pathlib.Path("example/base").resolve()
    assert settings.depletionSolver == "48"
    assert settings.rundir == pathlib.Path("example/run").resolve()
    assert not settings.useTempDir
    assert settings.fittingOrder == 0
    assert settings.numFittingPoints == 2
    assert not settings.unboundedFitting  # default

    serpent = settings.serpent

    assert serpent.acelib.parent == serpentdata
    assert serpent.acelib.name == "sss_endfb7u.xsdata"
    assert serpent.declib.parent == serpentdata
    assert serpent.declib.name == "sss_endfb71.dec"
    assert serpent.nfylib.parent == serpentdata
    assert serpent.nfylib.name == "sss_endfb71.nfy"

    assert serpent.particles == int(5e6)
    assert serpent.generationsPerBatch == 10
    assert serpent.active == 20
    assert serpent.inactive == 15
    assert serpent.seed == 123456
    assert serpent.k0 == 1.2

    assert serpent.executable == "sss2"
    assert serpent.omp == 16
    assert serpent.mpi == 2

    sfv = settings.sfv
    assert sfv.modes == 10
    assert sfv.modeFraction == 0.75
    assert sfv.densityCutoff == 1e-20


def test_emptyconfig(tmpdir):
    cfg = tmpdir / "bad.cfg"
    cfg.write("[DEFAULT]\nkey = value\n")

    with pytest.raises(KeyError):
        HydepSettings.fromFile(cfg, strict=True)

    with pytest.warns(UserWarning):
        HydepSettings.fromFile(cfg, strict=False)
