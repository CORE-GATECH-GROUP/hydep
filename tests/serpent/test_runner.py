import pytest
from hydep.serpent import SerpentRunner

@pytest.mark.serpent
def test_runner():

    r = SerpentRunner()
    assert r.executable is None
    assert r.numOMP == 1
    assert r.numMPI == 1

    r.executable = "sss2"
    assert r.executable == "sss2"

    r.numOMP = 20
    assert r.numOMP == 20
    r.numMPI = 20
    assert r.numMPI == 20

    with pytest.raises(TypeError):
        r.numOMP = 1.5

    with pytest.raises(ValueError):
        r.numOMP = 0

    with pytest.raises(ValueError):
        r.numOMP = -1

    with pytest.raises(TypeError):
        r.numMPI = 1.5

    with pytest.raises(ValueError):
        r.numMPI = 0

    with pytest.raises(ValueError):
        r.numMPI = -1

    r = SerpentRunner(executable="sss2", numOMP=20, numMPI=4)

    cmd = r.makeCmd()
    assert int(cmd[cmd.index("-omp") + 1]) == r.numOMP == 20

    assert cmd[0].startswith("mpi")
    assert r.numMPI == 4
    for sub in cmd[1:cmd.index(r.executable)]:
        if str(r.numMPI) == sub:
            break
    else:
        raise ValueError(f"Number of MPI tasks not found in {cmd}")
