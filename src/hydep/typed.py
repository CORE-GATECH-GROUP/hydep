from collections.abc import Iterable


class Descriptor:

    __slots__ = ("name", "value")

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype):
        return obj.__dict__[self.name]

    def __set__(self, obj, value):
        obj.__dict__[self.name] = value


class TypedAttr(Descriptor):
    __slots__ = Descriptor.__slots__ + ("types", "allowNone")

    def __init__(self, name, types, allowNone=False):
        if not isinstance(types, Iterable):
            types = types,
        self.types = types
        self.allowNone = bool(allowNone)
        super().__init__(name)

    def __set__(self, obj, value):
        if value is None and self.allowNone:
            pass
        elif not isinstance(value, self.types):
            raise TypeError(
                "Cannot set {}.{} to {} as is not {}".format(
                    obj.__class__.__name__, self.name, value,
                    ", or ".join(map(str, self.types)),
                )
            )
        super().__set__(obj, value)


class BoundedTyped(Descriptor):

    __slots__ = Descriptor.__slots__ + ("types", "le", "lt", "ge", "gt", "allowNone")

    def __init__(self, name, types, le=None, lt=None, ge=None, gt=None, allowNone=False):
        assert any(x is not None for x in {le, lt, ge, gt}), (name, le, lt, ge, gt)
        assert not (le is not None and lt is not None), (name, le, lt)
        assert not (ge is not None and gt is not None), (name, ge, gt)
        super().__init__(name)
        self.types = types
        self.le = le
        self.lt = lt
        self.ge = ge
        self.gt = gt
        self.allowNone = allowNone

    def __set__(self, obj, value):
        if value is None:
            if self.allowNone:
                return super().__set__(obj, None)
            raise TypeError(
                "Cannot set {}.{} to None".format(
                    obj.__class__.__name__, self.name))

        if not isinstance(value, self.types):
            raise TypeError(
                "Cannot set {}.{} to {} as is not {}".format(
                    obj.__class__.__name__, self.name, value, self.types
                )
            )
        if self.gt is not None and value <= self.gt:
            raise ValueError(
                "{}.{} must be > {}, not {}".format(
                    obj.__class__.__name__, self.name, self.gt, value
                )
            )
        elif self.ge is not None and value < self.ge:
            raise ValueError(
                "{}.{} must be >= {}, not {}".format(
                    obj.__class__.__name__, self.name, self.ge, value
                )
            )

        if self.lt is not None and value >= self.lt:
            raise ValueError(
                "{}.{} must be < {}, not {}".format(
                    obj.__class__.__name__, self.name, self.lt, value
                )
            )
        elif self.le is not None and value > self.le:
            raise ValueError(
                "{}.{} must be <= {}, not {}".format(
                    obj.__class__.__name__, self.name, self.le, value
                )
            )

        super().__set__(obj, value)


class IterableOf(Descriptor):

    __slots__ = Descriptor.__slots__ + ("types", "allowNone")

    def __init__(self, name, types, allowNone=False):
        super().__init__(name)
        self.types = types
        self.allowNone = bool(allowNone)

    def __set__(self, obj, value):
        if value is None and self.allowNone:
            pass
        elif not isinstance(value, Iterable):
            raise TypeError(
                "{}.{} must be iterable, not {}".format(
                    obj.__class__.__name__, self.name, type(value)
                )
            )
        elif not all(isinstance(x, self.types) for x in value):
            raise TypeError(
                "All elements of {}.{} must be {}".format(
                    obj.__class__.__name__, self.name, self.types
                )
            )
        super().__set__(obj, value)
