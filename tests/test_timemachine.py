import numpy
import pytest
import hydep
import hydep.internal


@pytest.fixture
def microxs():
    zai = (922350, 922350, 922380, 922380, 541350)
    rxn = (18, 102, 18, 102, 102)
    mxs = numpy.array([[0.02], [0.06], [0.05], [0.07], [1e3]])
    return hydep.internal.MicroXsVector.fromLongFormVectors(
        zai, rxn, mxs, assumeSorted=False
    )


def test_oneGroupReactionRates(microxs):
    times = [0, 100]
    incomingMicroXs = [[microxs], [microxs * 2]]
    expectedXsVector = microxs * 1.5

    timemachine = hydep.internal.XsTimeMachine(1, times, incomingMicroXs)

    flux = numpy.array([[4e6]])

    # Reaction rates are expected to be one group values
    # Computed by integrating microscopic cross sections and flux / volume
    # over energy
    # Quantities should be a single vector
    expectedRates = expectedXsVector.mxs[:, 0] * flux[0, 0]

    rxns = timemachine.getReactionRatesAt(0.5 * (times[0] + times[1]), flux)

    assert len(rxns) == 1
    assert rxns[0].zai == expectedXsVector.zai
    assert rxns[0].rxns == expectedXsVector.rxns
    assert rxns[0].mxs == pytest.approx(expectedRates)
