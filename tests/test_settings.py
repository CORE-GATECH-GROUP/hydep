import pytest
from hydep.settings import HydepSettings


def test_settings():
    ARCHIVE = "archive on success"
    DEP_SOLVER = "depletion solver"
    BC = "boundary conditions"

    h = HydepSettings()
    h.update(
        {ARCHIVE: "true", DEP_SOLVER: "cram48", BC: "reflective reflective vacuum"}
    )

    assert h.archiveOnSuccess
    assert h.depletionSolver == "cram48"
    assert h.boundaryConditions == ["reflective", "reflective", "vacuum"]

    h.update({ARCHIVE: "0", BC: "vacuum"})
    assert not h.archiveOnSuccess
    assert h.boundaryConditions == ["vacuum"] * 3

    with pytest.raises(TypeError, match=".*archive.*bool"):
        h.update({ARCHIVE: "positive"})
    assert not h.archiveOnSuccess

    with pytest.raises(ValueError, match=".*[B|b]oundary"):
        HydepSettings().update({BC: ["reflective", "very strange", "vacuum"]})
    assert h.boundaryConditions == ["vacuum"] * 3

    with pytest.raises(TypeError):
        h.archiveOnSuccess = 1

    fresh = HydepSettings(
        archiveOnSuccess=True,
        depletionSolver="testSolver",
        boundaryConditions="reflective",
    )
    assert fresh.boundaryConditions == ["reflective"] * 3
    assert fresh.archiveOnSuccess
    assert fresh.depletionSolver == "testSolver"
