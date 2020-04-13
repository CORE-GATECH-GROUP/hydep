"""Unit tests for SFV interface, not module"""
import pytest
from hydep.settings import SfvSettings, Settings


@pytest.mark.sfv
def test_config():
    """Test basic configuration, value checking, etc"""
    settings = SfvSettings(modes=1, modeFraction=0.5, densityCutoff=1e-6)

    assert settings.modes == 1
    assert settings.modeFraction == 0.5
    assert settings.densityCutoff == 1e-6

    with pytest.raises(ValueError):
        settings.modeFraction = 0

    with pytest.raises(ValueError):
        settings.modeFraction = 50

    assert settings.modeFraction == 0.5

    with pytest.raises(TypeError):
        settings.modes = 1.5

    with pytest.raises(ValueError):
        settings.modes = -1

    with pytest.raises(ValueError):
        settings.modes = 0

    assert settings.modes == 1

    settings.modes = None
    assert settings.modes is None

    with pytest.raises(ValueError):
        settings.densityCutoff = -1

    with pytest.raises(TypeError):
        settings.densityCutoff = None

    # Test user-friendly config
    settings.update(
        {"modes": "none", "mode fraction": "0.75", "density cutoff": "1E-2"}
    )

    assert settings.modes is None
    assert settings.modeFraction == 0.75
    assert settings.densityCutoff == 1e-2

    with pytest.raises(TypeError):
        settings.update({"modes": "1.7"})

    with pytest.raises(ValueError):
        settings.update({"mode fraction": "2.0"})

    with pytest.raises(ValueError):
        settings.update({"density cutoff": "-2"})

    settings.update({"modes": "200"})
    assert settings.modes == 200

    with pytest.raises(ValueError):
        settings.update({"modes": "200", "fake setting": "1"})


def test_fromSettings():
    """Test the integration into the dynamic settings framework"""
    hsettings = Settings()

    hsettings.updateAll(
        {
            "hydep": {},
            "hydep.sfv": {
                "modes": "1E6",
                "mode fraction": "0.5",
                "density cutoff": "1E-5",
            },
        }
    )
    assert hsettings.sfv.modes == 1e6
    assert hsettings.sfv.modeFraction == 0.5
    assert hsettings.sfv.densityCutoff == 1e-5
