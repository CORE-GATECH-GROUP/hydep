import numpy
import pytest
from hydep.internal.xs import (
    XsIndex,
    MaterialData,
    MaterialDataArray,
    DataBank,
)


@pytest.fixture
def xsInputs():
    index = XsIndex(
        [10010, 541350, 641550, 922350, 922380],
        [102, 102, 102, 102, 18, 18, 102],
        [0, 1, 2, 3, 5, 7],
    )

    data = numpy.array([1e-2, 1e3, 1e2, 0.06, 0.02, 0.05, 0.07])

    return MaterialData(index, data)


def test_xsfixture(xsInputs):
    index = xsInputs.index
    assert len(index) == 7
    assert index.zais == (10010, 541350, 641550, 922350, 922380)
    assert index.rxns == (102, 102, 102, 102, 18, 18, 102)
    assert index(10010, 102) == 0
    assert index(922350, 18) == 4
    assert index(922380, 102) == 6

    assert tuple(index.getReactions(10010)) == ((102, 0),)
    assert tuple(index.getReactions(541350)) == ((102, 1),)
    assert tuple(index.getReactions(641550)) == ((102, 2),)
    assert tuple(index.getReactions(922350)) == ((102, 3), (18, 4))
    assert tuple(index.getReactions(922380)) == ((18, 5), (102, 6))

    for ix in range(len(index)):
        zai, rxn = index[ix]
        assert index(zai, rxn) == ix

    for ix, (zai, rxn) in enumerate(index):
        assert index[ix] == (zai, rxn)

    assert xsInputs.data == pytest.approx(
        [1e-2, 1e3, 1e2, 0.06, 0.02, 0.05, 0.07]
    )

    assert xsInputs.getReactions(10010) == {102: 1e-2}
    assert xsInputs.getReactions(541350) == {102: 1e3}
    assert xsInputs.getReactions(641550) == {102: 1e2}
    assert xsInputs.getReactions(922350) == {18: 0.02, 102: 0.06}
    assert xsInputs.getReactions(922380) == {102: 0.07, 18: 0.05}

    # Non existent data
    assert xsInputs.getReactions(333) is None


def test_indexFailure():
    # Length of pointer and zai vectors are inconsistent
    with pytest.raises(ValueError, match=".*isotopes"):
        XsIndex(
            [10010, 922350], [102, 18], [0, 1, 2, 3],
        )
    with pytest.raises(ValueError, match=".*isotopes"):
        XsIndex(
            [10010, 922350], [102, 18], [0, 1],
        )

    # Length of reactions inconsistent with pointer vector
    with pytest.raises(ValueError, match=".*reactions"):
        XsIndex(
            [10010, 922350], [102, 18], [0, 2, 3],
        )

    valid = XsIndex([10010, 922350], [102, 102, 18], [0, 1, 3],)

    with pytest.raises(ValueError):
        valid.findZai(541350)

    with pytest.raises(ValueError):
        valid(10010, 18)


@pytest.fixture
def xsArray(xsInputs):
    return MaterialDataArray(
        xsInputs.index, numpy.array([xsInputs.data, xsInputs.data])
    )


def test_xsMath(xsArray):
    orig = xsArray.data.copy()

    add = xsArray + xsArray
    assert add.index is xsArray.index
    assert add.data == pytest.approx(orig * 2)

    add += xsArray
    assert add.data == pytest.approx(orig * 3)

    mul = xsArray * 2
    assert mul.data == pytest.approx(orig * 2)
    assert mul.index is xsArray.index

    mul *= 0.5
    assert mul.data == pytest.approx(orig)
    assert xsArray.data == pytest.approx(orig)

    assert (1.5 * xsArray).data == pytest.approx(orig * 1.5)

    div = xsArray / 2
    assert div.data == pytest.approx(orig / 2)

    div /= 2
    assert div.data == pytest.approx(orig / 4)

    linc = MaterialDataArray.fromLinearCombination(
        (0.5, xsArray), (1.5, xsArray)
    )
    assert linc.data == pytest.approx(orig * 2)

    assert xsArray.data == pytest.approx(orig)


@pytest.mark.parametrize("order", (0, 1, 2))
def test_extrapolation(xsArray, order):
    """Test the validity of the microxs extrapolation

    It is not impossible that a fit is requested that exceeds
    the maximum order based on the number of stored points.
    This is not mathematically optimal and may be unstable,
    but could be common for initial phases in the sequence.

    Rather than make an error, the process is to use the
    highest available order is used. If a single point is given, yet
    linear fitting is requested, constant fitting will be used until
    two or more points are added to the collection.

    This test is parametrized to study that behavior.

    """

    nmaterials = xsArray.data.shape[0]
    bank = DataBank(
        nsteps=2, nmaterials=nmaterials, rxnIndex=xsArray.index, order=order,
    )
    assert bank.reactionIndex is xsArray.index
    assert bank.nsteps == 2
    assert bank.nmaterials == nmaterials
    assert bank.nreactions == len(xsArray.index)
    assert bank.shape == (2, nmaterials, len(xsArray.index))
    assert bank.stacklen == 0  # nothing has been passed

    times = [0, 25]
    targetTime = 100
    scale = 2  # how to scale the second dataset y1 = scale * y0

    if order == 0:
        # Average between one and scale
        weight = 1.5
    else:
        # Linear extrapolation
        weight = 1 + (targetTime - times[0])*(scale - 1)/(times[1] - times[0])

    # Populate data
    bank.push(times[0], xsArray)
    assert bank.stacklen == 1

    bank.push(times[1], MaterialDataArray(xsArray.index, xsArray.data * scale))
    assert bank.stacklen == 2

    # Retrieve pushed data
    d0 = bank.at(times[0])
    assert d0.index is bank.reactionIndex
    assert d0.data == pytest.approx(xsArray.data)

    # Check consistency with original material-wise microscopic cross sections
    mt0 = d0[0]
    for zai in mt0.index.zais:
        reactions = mt0.getReactions(zai)
        for rxn, mxs in reactions.items():
            assert mxs == xsArray.data[0, xsArray.index(zai, rxn)]

    d1 = bank.at(times[1])
    assert d1.index is bank.reactionIndex
    assert d1.data == pytest.approx(xsArray.data * scale)
    mt1 = d1[0]
    for zai in mt1.index.zais:
        reactions = mt1.getReactions(zai)
        for rxn, mxs in reactions.items():
            assert mxs == xsArray.data[0, xsArray.index(zai, rxn)] * scale

    # Apply the fit
    extrap = bank.at(targetTime)
    assert extrap.index is bank.reactionIndex
    assert extrap.data == pytest.approx(xsArray.data * weight)

    # Check fit for reaction rates
    flux = numpy.array([[1E16], [4E16]], dtype=float)

    r0 = bank.getReactionRatesAt(times[0], flux)
    assert r0.data == pytest.approx(xsArray.data * flux)

    r1 = bank.getReactionRatesAt(times[1], flux)
    assert r1.data == pytest.approx(xsArray.data * flux * scale)

    rfit = bank.getReactionRatesAt(targetTime, flux)
    assert rfit.data == pytest.approx(xsArray.data * flux * weight)

    rf0 = rfit[0]
    for zai in rf0.index.zais:
        rates = rf0.getReactions(zai)
        for rxn, rate in rates.items():
            assert rate == pytest.approx(
                xsArray.data[0, xsArray.index(zai, rxn)] * weight * flux[0, 0]
            )
