import os
import pathlib
from collections import namedtuple
from enum import Enum, auto
import typing
import re


DataLibraries = namedtuple("DataLibraries", "xs decay nfy sab")
ProblematicIsotopes = namedtuple("ProblematicIsotopes", "missing replacements")
ProblematicIsotopes.__doc__ = """Simple container for problematic isotopes

Parameters
----------
missing : set of (int, int, int)
    Isotope ZAI triplet with number of protons, number of neutrons and
    neutrons, and metastable flag. Isotopes here were requested by
    the user but do not exist in the data library
replacements : dict
    Dictionary mapping ZAI triplets of requested isotopes to
    (Z, A) tuples of the new name. This is intended to help
    handle metastable isotopes that exist under different
    integer identifiers

"""


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


def findProblemIsotopes(
    stream, candidateZAIs: typing.Iterable[typing.Tuple[int, int, int]],
) -> ProblematicIsotopes:
    """Find isotopes that don't exist, or exist under new names

    Metastable isotopes have altered ZAI numbers in the Serpent xs
    file. For example, Am242_m1 is stored as 95342.

    Parameters
    ----------
    stream : readable
        Stream containing file data, like from opening the file
    candidateZAIs : iterable of (int, int, int)
        Isotopes ZAI identifiers that are likely to be used in the
        simulation

    Returns
    -------
    ProblematicIsotopes
        Containing information on isotopes that are in ``candidateZAIs``
        but not in the data file at all, or exist under a different
        name

    """
    reg = re.compile(r"\s+(\d{,6})\.\d{2}c\s+.*\.\d{2}c\s+\d\s+(\d{4,})\s+(\d+)")
    replacements = {}
    previous = set()
    candidates = set(candidateZAIs)

    line = stream.readline()
    # VER good candidate for python 3.8 use := operator
    while line:
        match = reg.match(line)
        if match is None:
            line = stream.readline()
            continue
        try:
            serpentZA, ZA, meta = match.groups()
            if (ZA, meta) in previous:
                line = stream.readline()
                continue

            previous.add((ZA, meta))
            z, a = divmod(int(ZA), 1000)
            zai = (z, a, int(meta))

            if zai not in candidates:
                line = stream.readline()
                continue
            if ZA != serpentZA:
                replacements[zai] = divmod(int(serpentZA), 1000)
        except Exception as ee:
            raise RuntimeError(f"Failed to process line\n{line}") from ee

        candidates.remove(zai)
        if not candidates:
            break
        line = stream.readline()

    return ProblematicIsotopes(missing=candidates, replacements=replacements)
