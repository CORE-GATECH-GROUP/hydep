import copy

import pytest
import numpy
import hydep
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


@pytest.mark.parametrize("applyThreshold", (True, False))
def test_compBundle(applyThreshold):
    fuel0 = hydep.BurnableMaterial("comp bundle 0", mdens=10.4)

    fuel0[922350] = 7e-4
    fuel0[80160] = 4e-2
    fuel0[922380] = 2e-2

    fuel1 = fuel0.copy()
    fuel1[942390] = 1e-6

    # add trace isotope
    TRACE_KEY = 541350
    TRACE_VALUE = 1e-7
    fuel0[TRACE_KEY] = TRACE_VALUE

    expIsotopes = set(fuel0).union(fuel1)

    if applyThreshold:
        for iso in fuel0:
            if iso.zai == TRACE_KEY:
                break
        else:
            raise ValueError(TRACE_KEY)
        expIsotopes.remove(iso)

    bundle = utils.compBundleFromMaterials(
        [fuel0, fuel1], threshold=TRACE_VALUE if applyThreshold else 0.0
    )

    assert len(bundle.isotopes) == len(expIsotopes)
    if applyThreshold:
        assert TRACE_KEY not in bundle.isotopes
    assert bundle.densities.shape == (2, len(bundle.isotopes))

    # check materials
    for isox, key in enumerate(bundle.isotopes):
        d0 = fuel0.get(key)
        assert bundle.densities[0, isox] == (0.0 if d0 is None else d0), key
        d1 = fuel1.get(key)
        assert bundle.densities[1, isox] == (0.0 if d1 is None else d1), key
