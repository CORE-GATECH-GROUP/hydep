"""Tests for some fission yield internals"""
import numpy
import pytest
from hydep.internal import FissionYield, FissionYieldDistribution


@pytest.fixture
def referenceDistribution():
    return {
        0.0253: {541350: 0.021, 621490: 0.04},
        500E3: {541350: 0.05, 621490: 0.03},
        14E6: {621490: 0.03, 541350: 0.06, 400960: 0.05},
    }


def test_fissionYieldDistribution(referenceDistribution):
    actual = FissionYieldDistribution(referenceDistribution)
    assert len(actual) == len(referenceDistribution)
    assert actual.energies == tuple(sorted(referenceDistribution))

    for key, value in actual.items():
        for product, fyield in value.items():
            # Creates some elements that are zeros, implied
            # by missing entries in an incoming fission yield matrix
            expected = referenceDistribution[key].get(product)
            if expected is None:
                assert fyield == pytest.approx(0)
            else:
                assert fyield == expected
            assert fyield == referenceDistribution[key].get(product, 0.0)
        assert value == actual[key]
        assert actual.get(key) == value

    for ix, ene in enumerate(actual):
        assert actual.energies[ix] == ene
        assert actual.at(ix) == actual[ene]
        assert ene in actual

    FAKE_ENERGY = -1.0
    assert actual.get(FAKE_ENERGY) is None

    ABS_TOL = 1E-3
    MIN_ENE = min(actual.energies)
    OFF_LOWER = MIN_ENE - ABS_TOL / 2
    lower = actual.get(OFF_LOWER, atol=ABS_TOL)
    assert lower is not None
    assert lower == actual[MIN_ENE]

    for value in actual.values():
        for _k, v in actual.items():
            if value == v:
                break
        else:
            raise ValueError(
                "Could not find a match using values generator")

    with pytest.raises(TypeError):
        actual.at(0.0253)


@pytest.fixture
def refFissionYields(referenceDistribution):
    ene = min(referenceDistribution)
    products = []
    yields = []
    for prod in sorted(referenceDistribution[ene]):
        products.append(prod)
        yields.append(referenceDistribution[ene][prod])
    return FissionYield(tuple(products), numpy.array(yields))


def test_singleEnergy(refFissionYields):
    origYields = refFissionYields.yields.copy()

    assert len(refFissionYields) == origYields.size

    for ix, prod in enumerate(refFissionYields.products):
        assert prod in refFissionYields
        assert refFissionYields[prod] == origYields[ix]

    for prod, yld in refFissionYields.items():
        assert refFissionYields[prod] == yld

    SCALAR = 2
    new = refFissionYields * SCALAR

    assert new.yields == pytest.approx(SCALAR * origYields)
    for key, value in new.items():
        assert value == pytest.approx(SCALAR * refFissionYields[key])

    new *= (1 / SCALAR)
    assert new.yields == pytest.approx(origYields)
    for key, value in new.items():
        assert value == pytest.approx(refFissionYields[key])

    added = refFissionYields + new
    assert added.yields == pytest.approx(SCALAR * origYields)
    for key, value in added.items():
        assert value == pytest.approx(SCALAR * refFissionYields[key])
