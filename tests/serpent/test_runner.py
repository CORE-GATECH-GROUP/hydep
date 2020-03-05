from unittest.mock import patch
import pytest
from hydep.serpent import SerpentRunner, SerpentSettings

MAGIC_OMP_THREADS = 1234

@pytest.mark.serpent
@patch.dict("os.environ", {"OMP_NUM_THREADS": str(MAGIC_OMP_THREADS)})
def test_runner():

    r = SerpentRunner()
    assert r.executable is None
    assert r.omp == MAGIC_OMP_THREADS
    assert r.mpi == 1

    r.executable = "sss2"
    assert r.executable == "sss2"

    r.omp = 20
    assert r.omp == 20
    r.mpi = 20
    assert r.mpi == 20

    with pytest.raises(TypeError):
        r.omp = 1.5

    with pytest.raises(ValueError):
        r.omp = 0

    with pytest.raises(ValueError):
        r.omp = -1

    with pytest.raises(TypeError):
        r.mpi = 1.5

    with pytest.raises(ValueError):
        r.mpi = 0

    with pytest.raises(ValueError):
        r.mpi = -1

    r = SerpentRunner(executable="sss2", omp=20, mpi=4)

    cmd = r.makeCommand()
    assert int(cmd[cmd.index("-omp") + 1]) == r.omp == 20

    assert cmd[0].startswith("mpi")
    assert r.mpi == 4
    for sub in cmd[1:cmd.index(r.executable)]:
        if str(r.mpi) == sub:
            break
    else:
        raise ValueError(f"Number of MPI tasks not found in {cmd}")

@pytest.mark.serpent
def test_config():
    settings = SerpentSettings(executable="sss2", omp="10", mpi="4")
    runner = SerpentRunner()

    runner.configure(settings)
    assert runner.executable == "sss2"
    assert runner.omp == 10
    assert runner.mpi == 4
