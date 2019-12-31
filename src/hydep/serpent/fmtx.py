"""Universe-based fission matrix parser

No guaruntee that universes are indexed in order according to
the indexing vector ``fmtx_uni``

Maybe eventually integrate this into / with ``serpentTools``?
"""

import re
from collections import namedtuple

import scipy.sparse


__all__ = ["FissionMatrixFile", "parseFmtx"]


FissionMatrixFile = namedtuple("FissionMatrixFile", "universes matrix")

UNI_REGEX = re.compile(r"fmtx_uni\s*\(\s*(\d+).*'(\d+)'")
ZEROS_REGEX = re.compile(r".* zeros\((\d+),(\d+)\)")
MATRIX_REGEX = re.compile(r"fmtx_t\s*\(\s*(\d+),\s*(\d+)\)\s*=\s*([0-9Ee\+-\.]+)")


def parseFmtx(stream):
    """Process a stream containing fission matrix data

    Parameters
    ----------
    stream : io.TextBase
        Readable stream of text data, like from an opened file

    Returns
    -------
    FissionMatrixFile
        Processed file contents. ``FissionMatrixFile.matrix``
        is the fission matrix, stored as a ``scipy.sparse.csrmatrix``
        Compressed Sparse Row (CSR). ``FissionMatrixFile.universes``
        is a tuple describing the universe ordering of the matrix.
        Universe ``u`` can be found with ``universes.index(u)``

    """
    # Find indexing vector

    match = None
    for line in stream:
        if not line or line[0] == "%":
            continue
        match = UNI_REGEX.match(line)
        if match is not None:
            break
    else:
        raise IOError("Could not find any fission matrix data")

    # Process index vector
    # Not likely sorted for....reasons?

    indexes = {}
    while line and match is not None:
        position, universe = match.groups()
        indexes[int(position)] = universe
        line = stream.readline()
        match = UNI_REGEX.match(line)

    if not line:
        raise EOFError("Failed to finish processing indexes")

    # look for matrix size

    match = ZEROS_REGEX.match(line)
    while line and match is None:
        line = stream.readline()
        match = ZEROS_REGEX.match(line)

    if not line:
        raise EOFError("Failed to find matrix shape")

    nrows, ncols = [int(x) for x in match.groups()]

    if nrows != ncols:
        raise ValueError("{} {}".format(nrows, ncols))

    # Read up to matrix population

    match = MATRIX_REGEX.match(line)
    while line and match is None:
        line = stream.readline()
        match = MATRIX_REGEX.match(line)

    if not line:
        raise EOFError("Failed to find matrix values")

    data = []
    rows = []
    cols = []

    while line and match is not None:
        row, col, value = match.groups()
        rows.append(int(row) - 1)
        cols.append(int(col) - 1)
        data.append(float(value))
        line = stream.readline()
        match = MATRIX_REGEX.match(line)

    fmtx = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(nrows, ncols))

    return FissionMatrixFile(tuple(indexes[k] for k in sorted(indexes)), fmtx.tocsr())
