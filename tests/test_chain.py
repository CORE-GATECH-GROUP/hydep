import math

import pytest
from hydep.constants import REACTION_MT_MAP
from hydep.internal import ReactionTuple, DecayTuple, getIsotope


def test_chain(simpleChain):
    assert len(simpleChain) == 17

    for ix, iso in enumerate(sorted(simpleChain)):
        assert simpleChain[ix] is iso
        assert simpleChain.index(iso) == ix
        assert (
            simpleChain.find(name=iso.name) is simpleChain.find(zai=iso.triplet) is iso
        )


def test_u5(simpleChain):
    reactionData = {
        ReactionTuple(REACTION_MT_MAP[t[0]], *t[1:])
        for t in (
            ["(n,2n)", getIsotope(name="U234"), 1.0, -5297781.0],
            ["(n,3n)", getIsotope(name="U233"), 1.0, -12142300.0],
            ["(n,4n)", getIsotope(name="U232"), 1.0, -17885600.0],
            ["(n,gamma)", getIsotope(name="U236"), 1.0, 6545200.0],
            ["fission", None, 1.0, 193405400.0],
        )
    }
    u5Reactions = {t.mt: t for t in reactionData}

    u5DecayModes = {
        DecayTuple(getIsotope("Th231"), "alpha", 0.999999999928),
    }

    u5Index = simpleChain.index("U235")
    assert "U235" in simpleChain
    assert 922350 in simpleChain
    u5 = simpleChain.find(name="U235")
    assert simpleChain[u5Index] is u5
    assert simpleChain.find(zai=922350) is u5
    assert simpleChain.find(zai=(92, 235, 0)) is u5
    assert simpleChain.find(zai=(92, 235)) is u5

    assert u5.decayConstant == pytest.approx(math.log(2) / 2.22102e16)
    assert len(u5.reactions) == len(u5Reactions)

    for rxn in u5.reactions:
        expected = u5Reactions.pop(rxn.mt)
        assert expected is not None
        assert rxn.target == expected.target
        assert rxn.branch == pytest.approx(expected.branch)
        assert rxn.Q == pytest.approx(expected.Q)

    assert not u5Reactions, "Mismatch in reactions. Missing {}".format(u5Reactions)
    assert u5.decayModes == u5DecayModes

    assert len(u5.fissionYields) == 3
    assert sorted(u5.fissionYields) == pytest.approx([0.0253, 5e5, 1.4e7])

    products = tuple(
        sorted(
            getIsotope(name=name).zai
            for name in ("Y91", "Zr91", "Zr93", "Zr95", "Zr96")
        )
    )
    expYields = [
        [0.0582783, 4.41969e-10, 0.0634629, 0.0650274, 0.0633924],
        [0.0573342, 2.00998e-10, 0.06254, 0.0643197, 0.0620232],
        [0.0482267, 1.84979e-07, 0.0519317, 0.0517353, 0.0520296],
    ]
    # all fission yields in this test chain are thermal
    for ene, fydist in zip(sorted(u5.fissionYields), expYields):
        assert u5.fissionYields[ene].products == products
        assert u5.fissionYields[ene].yields == pytest.approx(fydist)
