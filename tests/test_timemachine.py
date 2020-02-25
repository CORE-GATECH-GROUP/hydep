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


@pytest.mark.parametrize("grow", ("init", "append"))
@pytest.mark.parametrize("order", (0, 1, 2))
def test_oneGroupReactionRates(microxs, order, grow):
    """Test the validity of the microxs extrapolation

    It is not impossible that a time machine will be created
    using a single point in time, yet a linear or higher-order
    fitting scheme is requested. This is not mathematically optimal
    and may be unstable, but could be common for initial phases in
    the sequence.

    Rather than make an error, the process is to use the
    highest available order is used. If a single point is given, yet
    linear fitting is requested, constant fitting will be used until
    two or more points are added to the collection.

    This test is parametrized to study that behavior.

    """
    times = [0, 100]
    targetTime = 25
    scale = 2
    if order == 0:
        weight = 1.5
    else:
        weight = (
            (times[1] - targetTime + scale*(targetTime - times[0]))
            / (times[1] - times[0])
        )

    expectedXsVector = microxs * weight

    incomingMicroXs = [[microxs]]
    if grow == "init":
        incomingMicroXs.append([microxs * scale])
        timemachine = hydep.internal.XsTimeMachine(order, times, incomingMicroXs)
    else:
        timemachine = hydep.internal.XsTimeMachine(order, [times[0]], incomingMicroXs)
        timemachine.append(times[1], [microxs * scale])

    flux = numpy.array([[4e6]])

    # Reaction rates are expected to be one group values
    # Computed by integrating microscopic cross sections and flux / volume
    # over energy
    # Quantities should be a single vector
    expectedRates = expectedXsVector.mxs[:, 0] * flux[0, 0]

    with pytest.warns(None) as record:
        rxns = timemachine.getReactionRatesAt(targetTime, flux)

    assert len(record) == 0
    assert len(rxns) == 1
    assert rxns[0].zai == expectedXsVector.zai
    assert rxns[0].rxns == expectedXsVector.rxns
    assert rxns[0].mxs == pytest.approx(expectedRates)
