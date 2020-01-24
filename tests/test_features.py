import pytest
import hydep.internal.features as hdfeat


def test_features():
    """Test the feature capabilities"""
    s = hdfeat.FeatureCollection({hdfeat.FISSION_MATRIX})
    assert hdfeat.FISSION_MATRIX in s
    assert len(s) == len(s.features) == 1
    assert bool(s)

    o = hdfeat.FeatureCollection(
        {hdfeat.MICRO_REACTION_XS, hdfeat.HOMOG_GLOBAL}, {"ABS"}
    )
    assert hdfeat.MICRO_REACTION_XS in o
    assert "ABS" in o
    for ix, f in enumerate(o):
        assert f in o
        if ix >= len(o.features):
            assert f in o.macroXS
        else:
            assert f in o.features

    # Unions and subsets
    u = s.union(o)
    assert all(f in u.features for f in s.features)
    assert all(f in u.features for f in o.features)
    assert all(f in u for f in s)
    assert all(f in u for f in s)
    assert all(f in u.macroXS for f in s.macroXS)
    assert all(f in u.macroXS for f in o.macroXS)
    a = o.union(s)
    assert a == u
    assert hash(a) == hash(u)
    assert s.issubset(u)
    assert o.issubset(u)

    with pytest.raises(TypeError):
        hdfeat.FeatureCollection({"ABS"})

    assert not hdfeat.FeatureCollection()
