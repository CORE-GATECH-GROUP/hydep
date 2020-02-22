import io
import pathlib
from unittest.mock import patch

import pytest
from hydep.serpent.utils import Library, findLibraries, findProblemIsotopes


def _testDataLib(fileMap, referenceFiles):
    actualDataLib = findLibraries(*(fileMap.get(k) for k in Library))

    for attr, inputkey in [
        ["xs", Library.ACE],
        ["decay", Library.DEC],
        ["nfy", Library.NFY],
        ["sab", Library.SAB],
    ]:
        actual = getattr(actualDataLib, attr)
        ref = referenceFiles[inputkey]
        assert actual.is_file()
        assert actual.samefile(ref), (attr, inputkey)
        assert str(actual) == str(actual.resolve())


@pytest.mark.serpent
@patch.dict("os.environ", {"SERPENT_DATA": ""})
def test_dataLibraries(mockSerpentData):
    dataDir = mockSerpentData[Library.DATA_DIR]

    _testDataLib(mockSerpentData, mockSerpentData)

    # Test with just base files and data directory

    bases = {
        k: mockSerpentData[k].relative_to(dataDir)
        for k in [Library.ACE, Library.DEC, Library.NFY, Library.SAB]
    }

    with patch.dict(bases, {Library.DATA_DIR: dataDir}):
        _testDataLib(bases, mockSerpentData)

    with patch.dict("os.environ", {"SERPENT_DATA": str(dataDir)}):
        _testDataLib(bases, mockSerpentData)

    # Just bases provided with no data directory
    with pytest.raises(EnvironmentError):
        _testDataLib(bases, mockSerpentData)

    # Bases with a bad data directory
    fakeDir = pathlib.Path("this shouldn't exist")
    with pytest.raises(NotADirectoryError), patch.dict(
        bases, {Library.DATA_DIR: fakeDir}
    ):
        _testDataLib(bases, mockSerpentData)

    # Just bases with a data directory that does not contain files
    with pytest.raises(FileNotFoundError), patch.dict(
        bases, {Library.DATA_DIR: dataDir.parent}
    ):
        _testDataLib(bases, mockSerpentData)


def test_problemIsotopes(fakeXsDataStream):

    p = findProblemIsotopes(fakeXsDataStream, ((95, 242, 0), (95, 242, 1)))
    assert not p.missing
    assert p.replacements == {(95, 242, 1): (95, 342)}

    # Don't request metastable, add a bad isotope
    fakeXsDataStream.seek(0, io.SEEK_SET)
    bad = (1, 200, 0)
    p = findProblemIsotopes(fakeXsDataStream, ((95, 242, 0), bad))
    assert not p.replacements
    assert p.missing == set((bad,))
