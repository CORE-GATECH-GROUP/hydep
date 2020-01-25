from collections import namedtuple

import pytest
import numpy
from hydep.internal.microxs import MicroXsVector, TemporalMicroXs

XsInputs = namedtuple("XsInputs", ["zai", "rxn", "mxs"])
ExpectedXsAttrs = namedtuple("ExpectedXsAttrs", ["zai", "zptr", "rxn", "mxs"])


@pytest.fixture
def microxsDict():
    out = {
        (922350, 18): numpy.array([0.02]),
        (541350, 102): numpy.array([1e3]),
        (922350, 102): numpy.array([0.06]),
        (922380, 18): numpy.array([0.05]),
        (10010, 102): numpy.array([1e-2]),
        (641550, 102): numpy.array([1e2]),
        (922380, 102): numpy.array([0.07]),
    }
    return out


@pytest.fixture
def expectedXsVectors(microxsDict):
    expectedZ = []
    expectedR = []
    expectedM = []
    expectedZptr = [0]

    ix = 0
    for (z, r), m in sorted(microxsDict.items(), key=lambda k: k[0]):

        if ix == 0:
            expectedZ.append(z)
        elif z != expectedZ[-1]:
            expectedZ.append(z)
            expectedZptr.append(ix)
        ix += 1
        expectedR.append(r)
        expectedM.append(m)

    expectedZptr.append(ix)

    expectedZ = numpy.array(expectedZ, dtype=int)
    expectedZptr = numpy.array(expectedZptr, dtype=int)
    expectedR = numpy.array(expectedR, dtype=int)
    expectedM = numpy.array(expectedM, dtype=float)

    return ExpectedXsAttrs(expectedZ, expectedZptr, expectedR, expectedM)


@pytest.fixture
def microXsInputs(microxsDict):
    zai = []
    rxn = []
    mxs = []
    for (z, r), m in microxsDict.items():
        zai.append(z)
        rxn.append(r)
        mxs.append(m)

    zai = numpy.array(zai)
    rxn = numpy.array(rxn)
    mxs = numpy.array(mxs)

    return XsInputs(zai, rxn, mxs)


def test_longform(microxsDict, expectedXsVectors, microXsInputs):

    mxsVector = MicroXsVector.fromLongFormVectors(
        microXsInputs.zai, microXsInputs.rxn, microXsInputs.mxs, assumeSorted=False
    )

    assert len(mxsVector) == len(microxsDict)

    assert (mxsVector.zai == expectedXsVectors.zai).all()
    assert (mxsVector.rxns == expectedXsVectors.rxn).all()
    assert (mxsVector.mxs == expectedXsVectors.mxs).all()
    assert len(mxsVector.zptr) == len(mxsVector.zai) + 1

    # test iteration

    for z, r, m in mxsVector:
        assert microxsDict[z, r] == m, (z, r)
        assert mxsVector.getReaction(z, r) == m

    # test z pointer
    allZ = list(k[0] for k in microxsDict)
    orderedZ = sorted(set(allZ))

    for ix, z in enumerate(orderedZ):
        nValues = allZ.count(z)
        assert mxsVector.zptr[ix + 1] - mxsVector.zptr[ix] == nValues, (z, nValues)

    orig = mxsVector.mxs.copy()

    mult = mxsVector * 2
    assert mult.mxs == pytest.approx(orig * 2)
    mult *= 2
    assert mult.mxs == pytest.approx(orig * 4)
    assert mxsVector.mxs == pytest.approx(orig)

    rxnMap = mxsVector.getReactions(922380)
    assert rxnMap
    expectedNum = len([k for k in microxsDict if k[0] == 922380])
    assert len(rxnMap) == expectedNum
    for key, value in rxnMap.items():
        assert (value == microxsDict[922380, key]).all(), key

    assert mxsVector.getReactions(-1) is None
    assert mxsVector.getReactions(-1, default=False) is False
    assert mxsVector.getReaction(-1, 18) is None
    assert mxsVector.getReaction(922380, -1) is None


@pytest.mark.parametrize("grow", ["init", "insert", "append", "reversed"])
def test_temporal(expectedXsVectors, grow):
    time = [0, 1]
    mxs = expectedXsVectors.mxs, expectedXsVectors.mxs * 2

    if grow == "init":
        tMxs = TemporalMicroXs(
            expectedXsVectors.zai,
            expectedXsVectors.zptr,
            expectedXsVectors.rxn,
            time=time,
            mxs=mxs,
            order=1,
        )
    elif grow == "insert":
        tMxs = TemporalMicroXs(
            expectedXsVectors.zai,
            expectedXsVectors.zptr,
            expectedXsVectors.rxn,
            order=1,
        )
        for t, m in zip(time, mxs):
            tMxs.insert(t, m)
    elif grow == "append":
        tMxs = TemporalMicroXs(
            expectedXsVectors.zai,
            expectedXsVectors.zptr,
            expectedXsVectors.rxn,
            order=1,
        )
        for t, m in zip(time, mxs):
            tMxs.append(t, m)
    elif grow == "reversed":
        tMxs = TemporalMicroXs(
            expectedXsVectors.zai,
            expectedXsVectors.zptr,
            expectedXsVectors.rxn,
            order=1,
        )
        for t, m in zip(reversed(time), reversed(mxs)):
            tMxs.insert(t, m)
    else:
        raise ValueError(grow)

    interpTime = 0.5 * (time[1] + time[0])
    interpXs = 0.5 * (mxs[1] + mxs[0])

    actual = tMxs(interpTime)

    assert (actual.zai == tMxs.zai).all()
    assert (actual.rxns == tMxs.rxns).all()
    assert (actual.zptr == tMxs.zptr).all()
    assert actual.mxs == pytest.approx(interpXs)
