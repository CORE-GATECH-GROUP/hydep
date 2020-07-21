from unittest.mock import patch
import pathlib

import pytest
from hydep.settings import Settings

from hydep.settings import SerpentSettings
hdserpent = pytest.importorskip("hydep.serpent")
from hydep.serpent.utils import Library


@pytest.fixture
def cleanEnviron():
    with patch.dict("os.environ", {"SERPENT_DATA": "", "OMP_NUM_THREADS": ""}):
        yield


@pytest.mark.serpent
@pytest.mark.parametrize(
    "attribute", ("particles", "active", "inactive", "generationsPerBatch", "mpi")
)
def test_integers(cleanEnviron, attribute):
    fresh = SerpentSettings()
    assert getattr(fresh, attribute) is None
    setattr(fresh, attribute, 10)
    assert getattr(fresh, attribute) == 10

    fkwargs = SerpentSettings(**{attribute: 10})
    assert getattr(fkwargs, attribute) == 10

    with pytest.raises(ValueError):
        setattr(fresh, attribute, -10)

    with pytest.raises(ValueError):
        SerpentSettings(**{attribute: -10})

    with pytest.raises(ValueError):
        setattr(fresh, attribute, 0)

    with pytest.raises(ValueError):
        SerpentSettings(**{attribute: 0})

    with pytest.raises(TypeError):
        setattr(fresh, attribute, 1.45)

    with pytest.raises(TypeError):
        SerpentSettings(**{attribute: 1.45})

    with pytest.raises(TypeError):
        setattr(fresh, attribute, "10")


@pytest.mark.serpent
def test_datafiles(cleanEnviron, mockSerpentData):
    fresh = SerpentSettings()

    assert fresh.datadir is None
    for attr in ["acelib", "declib", "nfylib", "sab"]:
        assert getattr(fresh, attr) is None, attr

    for attr, key in [
        ("acelib", Library.ACE),
        ("declib", Library.DEC),
        ("nfylib", Library.NFY),
        ("sab", Library.SAB),
    ]:
        setattr(fresh, attr, mockSerpentData[key])
        assert getattr(fresh, attr).samefile(mockSerpentData[key])
        assert getattr(fresh, attr).is_absolute()

    fresh.datadir = mockSerpentData[Library.DATA_DIR]
    assert fresh.datadir.is_dir()

    full = SerpentSettings(
        datadir=mockSerpentData[Library.DATA_DIR],
        acelib=mockSerpentData[Library.ACE].name,
        declib=mockSerpentData[Library.DEC].name,
        nfylib=mockSerpentData[Library.NFY].name,
    )

    # Exclude S(a,b) because it lives in a sub-directory
    for attr, key in [
        ("acelib", Library.ACE),
        ("declib", Library.DEC),
        ("nfylib", Library.NFY),
    ]:
        setattr(fresh, attr, mockSerpentData[key].name)
        assert getattr(fresh, attr).samefile(mockSerpentData[key])
        assert getattr(fresh, attr).is_absolute()
        assert getattr(full, attr).is_absolute()
        assert getattr(fresh, attr).samefile(getattr(full, attr))


@pytest.mark.serpent
def test_environ(cleanEnviron):
    FAKE_DIR = pathlib.Path(__file__).parent
    FAKE_OMP = 123

    with patch.dict(
        "os.environ", {"SERPENT_DATA": str(FAKE_DIR), "OMP_NUM_THREADS": str(FAKE_OMP)}
    ):
        bare = SerpentSettings()

    assert bare.datadir == FAKE_DIR
    assert bare.omp == FAKE_OMP


@pytest.mark.serpent
@pytest.mark.parametrize("useDataDir", (True, False))
@pytest.mark.parametrize("allStrings", (True, False))
@pytest.mark.parametrize("onHydep", (False, True))
def test_update(cleanEnviron, mockSerpentData, useDataDir, allStrings, onHydep):
    SETTINGS = {
        "acelib": mockSerpentData[Library.ACE],
        "nfylib": mockSerpentData[Library.NFY],
        "declib": mockSerpentData[Library.DEC],
        "thermal scattering": mockSerpentData[Library.SAB],
        "executable": "sss2",
        "mpi": 8,
        "omp": 64,
        "generations per batch": 10,
        "particles": 1000,
        "active": 50,
        "inactive": 20,
        "k0": 1.2,
        "fpy mode": "weighted",
        "fsp inactive batches": 5,
    }

    if useDataDir:
        for key in {"acelib", "declib", "nfylib"}:
            SETTINGS[key] = SETTINGS[key].name
        SETTINGS["datadir"] = mockSerpentData[Library.DATA_DIR]

    if allStrings:
        SETTINGS = {k: str(v) for k, v in SETTINGS.items()}

    if onHydep:
        SETTINGS = {"hydep": {}, "hydep.serpent": SETTINGS}
        hset = Settings()
        hset.updateAll(SETTINGS)
        serpent = hset.serpent
    else:
        serpent = SerpentSettings()
        serpent.update(SETTINGS)

    if useDataDir:
        assert serpent.datadir == mockSerpentData[Library.DATA_DIR]
    else:
        assert serpent.datadir is None
    assert serpent.acelib.samefile(mockSerpentData[Library.ACE])
    assert serpent.declib.samefile(mockSerpentData[Library.DEC])
    assert serpent.nfylib.samefile(mockSerpentData[Library.NFY])
    assert serpent.sab.samefile(mockSerpentData[Library.SAB])
    assert serpent.executable == "sss2"
    assert serpent.mpi == 8
    assert serpent.omp == 64
    assert serpent.generationsPerBatch == 10
    assert serpent.particles == 1000
    assert serpent.active == 50
    assert serpent.inactive == 20
    assert serpent.k0 == 1.2
    assert serpent.fpyMode == "weighted"
    assert serpent.fspInactiveBatches == 5
