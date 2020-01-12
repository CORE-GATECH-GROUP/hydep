import copy

import pytest
import numpy
from hydep.internal import utils


def fakeSeqInner(original, sequence, expcounts):
    for ix, value in enumerate(sequence):
        assert ix < expcounts
        assert value == pytest.approx(original, rel=0, abs=0)
        assert sequence[ix] == pytest.approx(original, rel=0, abs=0)


@pytest.mark.parametrize("dtype", (list, tuple, numpy.array))
def test_fakeSequence(dtype):

    data = dtype(range(10))
    original = copy.copy(data)

    counts = 10

    sequence = utils.FakeSequence(data, counts)
    assert len(sequence) == counts

    fakeSeqInner(original, sequence, counts)

    # Do it again as itertools.repeat (an alternative) only supports
    # one iteration

    fakeSeqInner(original, sequence, counts)
