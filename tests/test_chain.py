import math

import pytest
from hydep.internal import ReactionTuple, DecayTuple, getIsotope
from hydep.internal.symbols import REACTION_MTS


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
        ReactionTuple(REACTION_MTS[t[0]], *t[1:])
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
