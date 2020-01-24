import numbers
import pytest
import hydep.typed as htypes


class Typed:
    integer = htypes.TypedAttr("integer", numbers.Integral)
    integerOrNone = htypes.TypedAttr(
        "integerOrNone", numbers.Integral, allowNone=True)
    mixed = htypes.TypedAttr("mixed", (numbers.Integral, str))
    iterable = htypes.IterableOf("iterable", numbers.Integral)
    iterableOrNone = htypes.IterableOf(
        "iterableOrNone", numbers.Integral, allowNone=True
    )
    mixedIterable = htypes.IterableOf("mixedIterable", (numbers.Integral, str))


def test_typed():
    """Test for the internal type-helpers"""
    # Yay for dynamic languages!

    t = Typed()

    for attr, value in [
        ["integer", 1],
        ["integerOrNone", 1],
        ["integerOrNone", None],
        ["mixed", 1],
        ["mixed", "hello"],
        ["iterable", [1, 2, 3]],
        ["iterableOrNone", [1, 2, 3]],
        ["iterableOrNone", None],
        ["mixedIterable", [1, 2, "Hello", 3]],
    ]:
        assert attr in t.__class__.__dict__
        setattr(t, attr, value)
        assert getattr(t, attr) == value


def test_badTyped():

    t = Typed()

    with pytest.raises(TypeError, match=r".*Typed\.integer"):
        t.integer = 4.3

    with pytest.raises(TypeError, match=r".*Typed\.integer"):
        t.integer = None

    with pytest.raises(TypeError, match=r".*Typed\.mixed"):
        t.mixed = [1, 2, 3]

    with pytest.raises(TypeError, match=r".*Typed\.iterable"):
        t.iterable = 1

    with pytest.raises(TypeError, match=r".*Typed\.iterable"):
        t.iterable = None

    with pytest.raises(TypeError, match=r".*Typed\.iterable"):
        t.iterable = [1, 2, 3.4]


class Bounded:
    """Values are bounded between zero and one"""

    lower = htypes.BoundedTyped("lower", numbers.Real, gt=0.0)
    eqLower = htypes.BoundedTyped("eqLower", numbers.Real, ge=0.0)
    upper = htypes.BoundedTyped("upper", numbers.Real, lt=1.0)
    eqUpper = htypes.BoundedTyped("eqUpper", numbers.Real, le=1.0)
    both = htypes.BoundedTyped("both", numbers.Real, lt=1.0, gt=0.0)
    eqBoth = htypes.BoundedTyped("eqBoth", numbers.Real, le=1.0, ge=0.0)


def test_badBounded():

    with pytest.raises(AssertionError, match="none.*None.*None.*None.*None"):
        htypes.BoundedTyped("none", numbers.Real)

    with pytest.raises(AssertionError, match="le_and_lt"):
        htypes.BoundedTyped("le_and_lt", numbers.Real, le=0.0, lt=1e-5)

    with pytest.raises(AssertionError, match="ge_and_gt"):
        htypes.BoundedTyped("ge_and_gt", numbers.Real, ge=0.0, gt=1e-5)

    b = Bounded()
    for attr, value in [
        ["lower", 0.5],
        ["eqLower", 0.0],
        ["upper", 0.5],
        ["eqUpper", 1.0],
        ["both", 0.5],
        ["eqBoth", 1.0],
        ["eqBoth", 0.0],
    ]:
        assert attr in b.__class__.__dict__
        setattr(b, attr, value)
        assert getattr(b, attr) == value

    with pytest.raises(TypeError, match="Bounded.lower"):
        b.lower = "0.5"

    with pytest.raises(ValueError, match="Bounded.lower"):
        b.lower = -1.0

    with pytest.raises(ValueError, match="Bounded.eqLower"):
        b.eqLower = -1.0

    with pytest.raises(ValueError, match="Bounded.upper"):
        b.upper = 1.5

    with pytest.raises(ValueError, match="Bounded.eqUpper"):
        b.eqUpper = 1.5

    with pytest.raises(ValueError, match="Bounded.eqBoth"):
        b.eqBoth = 1.5

    with pytest.raises(ValueError, match="Bounded.eqBoth"):
        b.eqBoth = -1.0
