import math

import pytest
from hydep import DepletionChain
from hydep.internal import ReactionTuple, DecayTuple, getIsotope
from hydep.internal.symbols import REACTION_MTS


@pytest.fixture(scope="session")
def chain(tmpdir_factory):
    chainfile = tmpdir_factory.mktemp("chains").join("simple_chain.xml")
    with open(chainfile, "w") as stream:
        stream.write(
            """<depletion_chain>
  <nuclide name="Xe135" half_life="32904.0" decay_modes="1" decay_energy="567980.1000000001" reactions="5">
    <decay type="beta-" target="Cs135" branching_ratio="1.0"/>
    <reaction type="(n,2n)" Q="-6457100.0" target="Xe134"/>
    <reaction type="(n,3n)" Q="-14996200.0" target="Xe133"/>
    <reaction type="(n,gamma)" Q="7990390.0" target="Xe136"/>
    <reaction type="(n,p)" Q="-1927530.0" target="I135"/>
    <reaction type="(n,a)" Q="4361190.0" target="Te132"/>
  </nuclide>
  <nuclide name="U235" half_life="2.22102e+16" decay_modes="2" decay_energy="4619192.11" reactions="5">
    <!-- Intentionally skip sf decay <decay type="sf" target="U235" branching_ratio="7.2e-11"/> -->
    <decay type="alpha" target="Th231" branching_ratio="0.999999999928"/>
    <reaction type="(n,2n)" Q="-5297781.0" target="U234"/>
    <reaction type="(n,3n)" Q="-12142300.0" target="U233"/>
    <reaction type="(n,4n)" Q="-17885600.0" target="U232"/>
    <reaction type="(n,gamma)" Q="6545200.0" target="U236"/>
    <reaction type="fission" Q="193405400.0"/>
  </nuclide>
  <nuclide name="Am241" half_life="13651800000.0" decay_modes="2" decay_energy="5627985.35" reactions="6">
    <!-- Intentionally skip sf decay <decay type="sf" target="Am241" branching_ratio="4.3e-12"/> -->
    <decay type="alpha" target="Np237" branching_ratio="0.9999999999957"/>
    <reaction type="(n,2n)" Q="-6638281.0" target="Nothing"/>
    <reaction type="(n,3n)" Q="-12596700.0" target="Nothing"/>
    <reaction type="(n,4n)" Q="-19703000.0" target="Nothing"/>
    <reaction type="fission" Q="201960100.0"/>
    <reaction type="(n,gamma)" Q="5539101.0" target="Am242" branching_ratio="0.919"/>
    <reaction type="(n,gamma)" Q="5539101.0" target="Am242_m1" branching_ratio="0.081"/>
  </nuclide>
</depletion_chain>"""
        )
    yield DepletionChain.fromXml(str(chainfile))


@pytest.fixture
def u5Reactions():
    data = set(
        ReactionTuple(REACTION_MTS[t[0]], *t[1:])
        for t in (
            ["(n,2n)", getIsotope(name="U234"), 1.0, -5297781.0],
            ["(n,3n)", getIsotope(name="U233"), 1.0, -12142300.0],
            ["(n,4n)", getIsotope(name="U232"), 1.0, -17885600.0],
            ["(n,gamma)", getIsotope(name="U236"), 1.0, 6545200.0],
            ["fission", None, 1.0, 193405400.0],
        )
    )

    return {t.mt: t for t in data}


@pytest.fixture
def u5DecayModes():
    data = set(
        DecayTuple(getIsotope("Th231"), "alpha", 1.0),
    )
    return data


def test_chain(chain):
    assert len(chain) == 17

    for ix, iso in enumerate(sorted(chain)):
        assert chain[ix] is iso
        assert chain.index(iso) == ix
        assert chain.find(name=iso.name) is chain.find(zai=iso.triplet) is iso


def test_u5(chain, u5Reactions, u5DecayModes):

    u5Index = chain.index("U235")
    assert "U235" in chain
    assert 922350 in chain
    u5 = chain.find(name="U235")
    assert chain[u5Index] is u5
    assert chain.find(zai=922350) is u5
    assert chain.find(zai=(92, 235, 0)) is u5
    assert chain.find(zai=(92, 235)) is u5

    assert u5.decayConstant == pytest.approx(math.log(2) / 2.22102e16)
    assert len(u5.reactions) == len(u5Reactions)

    for rxn in u5.reactions:
        expected = u5Reactions.pop(rxn.mt)
        assert expected is not None
        assert rxn.target == expected.target
        assert rxn.branch == pytest.approx(expected.branch)
        assert rxn.Q == pytest.approx(expected.Q)

    assert not u5Reactions, "Mismatch in reactions. Missing {}".format(u5Reactions)
