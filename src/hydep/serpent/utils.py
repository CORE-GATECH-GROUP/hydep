import os
import pathlib
from collections import namedtuple
from enum import Enum, auto


DataLibraries = namedtuple("DataLibraries", "xs decay nfy sab")


class Library(Enum):
    ACE = auto()
    DEC = auto()
    NFY = auto()
    SAB = auto()
    DATA_DIR = auto()


def findLibraries(acelib, declib, nfylib, sab=None, datadir=None) -> DataLibraries:
    """Attempt to determine where the data libraries are located.

    The following modes are supported.

    1. Files are given as full paths
    2. Files are given as base names and exist in ``data directory``
    3. Files are given as base names and exist in the ``$SERPENT_DATA``
       environment variable

    Parameters
    ----------
    acelib : pathlib.Path
        Full path or basename of the primary cross section look up
        file
    declib : pathlib.Path
        Full path or basename of the decay reaction file
    nfylib : pathlib.Path
        Full path or basename of the neutron induced fission yield file
    sab : pathlib.Path, optional
        Full path to primary thermal scattering file. If not given,
        will be found using ``datadir`` semantics
    datadir : pathlib.Path, optional
        Common directory for data libraries. This will be used to
        expand any files that are given as short names. If ``acelib``
        does not exist as given, it will expanded to
        ``datadir / acelib``. Similarly, if ``sab`` is not given,
        it will be set as ``datadir / acedata / sssth1``

    Returns
    -------
    DataLibraries

    """
    nonexist = {}
    files = {}
    (files if acelib.is_file() else nonexist)[Library.ACE] = acelib
    (files if declib.is_file() else nonexist)[Library.DEC] = declib
    (files if nfylib.is_file() else nonexist)[Library.NFY] = nfylib
    if sab is not None and sab.is_file():
        files[Library.SAB] = sab
    else:
        nonexist[Library.SAB] = sab

    if not nonexist:
        # Link with lower return
        return DataLibraries(
            xs=files[Library.ACE].resolve(),
            decay=files[Library.DEC].resolve(),
            nfy=files[Library.NFY].resolve(),
            sab=files[Library.SAB].resolve(),
        )

    if datadir is None:
        datadir = os.environ.get("SERPENT_DATA")
        if datadir is None:
            raise EnvironmentError(
                "Need data directory or SERPENT_DATA environment "
                "variable to recover missing files"
            )
        datadir = pathlib.Path(datadir)
    if not datadir.is_dir():
        raise NotADirectoryError(f"{datadir} is not a directory")

    if sab is None:
        sab = datadir / "acedata" / "sssth1"
        if not sab.is_file():
            raise FileNotFoundError(
                f"Thermal scattering library not given and {sab} is not a file"
            )
        files[Library.SAB] = sab
        nonexist.pop(Library.SAB)

    for k, v in nonexist.items():
        candidate = datadir / v
        if candidate.is_file():
            files[k] = candidate
        else:
            raise FileNotFoundError(
                f"File {v} does not exist, and does not exist in {datadir}"
            )

    # Link with upper return
    return DataLibraries(
        xs=files[Library.ACE].resolve(),
        decay=files[Library.DEC].resolve(),
        nfy=files[Library.NFY].resolve(),
        sab=files[Library.SAB].resolve(),
    )
