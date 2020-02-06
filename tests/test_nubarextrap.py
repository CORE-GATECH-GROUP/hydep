import numpy
from numpy.polynomial.polynomial import polyfit, polyval
import pytest
from hydep.sfv.utils import NubarPolyFit


@pytest.mark.parametrize("order", (1, 2))
@pytest.mark.parametrize("how", ["insort", "append", "extend"])
def test_sfvnubar(how, order):
    # Linear function of time
    N_ITEMS = 5
    TEST_VECTOR = numpy.arange(N_ITEMS)
    TIMES = numpy.arange(1, 4)
    data = [TEST_VECTOR * t for t in TIMES]

    nubarExtrap = NubarPolyFit(order=order)
    if how == "insort":
        for t, d in zip(TIMES, data):
            nubarExtrap.insort(t, d)
    elif how == "append":
        for t, d in zip(TIMES, data):
            nubarExtrap.append(t, d)
    elif how == "extend":
        nubarExtrap.extend(TIMES, data)
    else:
        raise ValueError(how)

    assert len(nubarExtrap) == len(TIMES)
    for ix, d in enumerate(data):
        assert (nubarExtrap[ix] == d).all()
        assert (nubarExtrap(TIMES[ix]) == d).all()

    target = TIMES[0] + 0.5*TIMES[1]
    extrap = nubarExtrap(target)
    if order == 1:
        assert extrap == pytest.approx(data[0] + 0.5 * data[1])
    elif order == 2:
        for ix, row in enumerate(numpy.array(data).T):
            fit = polyfit(TIMES, row, order)
            assert extrap[ix] == pytest.approx(polyval(target, fit))
    else:
        raise ValueError(order)
