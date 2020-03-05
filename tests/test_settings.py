import random
import string

import pytest
from hydep.settings import HydepSettings, SubSetting


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
    assert h.boundaryConditions == ["reflective", "reflective", "vacuum"]
    assert h.fittingOrder == 2
    assert h.numFittingPoints == 4
    assert not h.unboundedFitting

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
    assert h.boundaryConditions == ["vacuum"] * 3
    assert h.fittingOrder == 1
    assert h.numFittingPoints is None
    assert h.unboundedFitting

    with pytest.raises(TypeError, match=".*archive.*bool"):
        h.update({ARCHIVE: "positive"})
    assert not h.archiveOnSuccess

    with pytest.raises(ValueError, match=".*[B|b]oundary"):
        HydepSettings().update({BC: ["reflective", "very strange", "vacuum"]})
    assert h.boundaryConditions == ["vacuum"] * 3

    with pytest.raises(TypeError):
        h.archiveOnSuccess = 1

    with pytest.raises(TypeError):
        h.unboundedFitting = 1

    fresh = HydepSettings(
        archiveOnSuccess=True,
        depletionSolver="testSolver",
        boundaryConditions="reflective",
    )
    assert fresh.boundaryConditions == ["reflective"] * 3
    assert fresh.archiveOnSuccess
    assert fresh.depletionSolver == "testSolver"

    # Test some conversion
    with pytest.raises(TypeError, match=".*bool.*integer"):
        fresh.asInt("test", False)


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
                self.truth = self.asBool("truth", v)

    t = settings.test
    assert isinstance(t, MySubSettings)
    assert not hasattr(settings, randomSection)
    assert t.truth

    settings.updateAll({"hydep": {}, "hydep.test": {"truth": "0"}})
    assert not t.truth

    fresh = HydepSettings()
    fresh.updateAll(
        {"hydep": {"depletion solver": "cram48"}, "hydep.test": {"truth": "n"}}
    )
    assert fresh.depletionSolver == "cram48"
    assert not fresh.test.truth

    with pytest.raises(ValueError, match=f".*{randomSection}"):
        fresh.updateAll({"hydep": {}, f"hydep.{randomSection}": {"key": "value"}})

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
