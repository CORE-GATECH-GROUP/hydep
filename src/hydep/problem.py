"""
Primary class for handling geometry and material information
"""

from hydep.lib import HighFidelitySolver, ReducedOrderSolver, BaseStore
from hydep import Model, Manager
from hydep.typed import TypedAttr
from hydep.internal import TimeStep, configmethod


class Problem(object):

    model = TypedAttr("model", Model)
    hf = TypedAttr("hf", HighFidelitySolver)
    rom = TypedAttr("rom", ReducedOrderSolver)
    dep = TypedAttr("dep", Manager)

    def __init__(self, model, hf, rom, dep, store=None):
        self._locked = False
        hf.checkCompatibility(dep)
        hf.checkCompatibility(rom)
        self.model = model
        self.hf = hf
        self.rom = rom
        self.dep = dep
        self.store = store

    @property
    def store(self):
        return self._store

    @store.setter
    def store(self, value):
        if self._locked:
            raise AttributeError(f"{self} is locked and should not be modified")
        if value is None:
            self._store = None
            return
        if not isinstance(value, BaseStore):
            raise TypeError(f"Store should be subclass of {BaseStore}, not {value}")
        self._store = value

    def beforeMain(self):
        self.dep.beforeMain(self.model)
        self.hf.beforeMain(self.model, self.dep.burnable, self.dep.chain)
        self.rom.beforeMain(self.model, self.dep.burnable, self.dep.chain)

        if self.store is None:
            from .h5store import HdfStore

            self.store = HdfStore.fromManager(self.dep, nGroups=1)

        self.store.beforeMain(
            tuple(self.dep.chain), [(i, m) for i, m in enumerate(self.dep.burnable)],
        )

    def solve(self):
        """Here we gooooooooooo"""
        time = TimeStep()
        self.hf.setHooks(self.dep.needs.union(self.rom.needs))
        self.beforeMain()
        result = None

    @configmethod
    def configure(self, config):
        """Configure each of the solvers

        Parameters
        ----------
        config : Union[ConfigParser, str, Path, Mapping]
            An object that is compatible with
            :class:`configparser.ConfigParser`. If a string or
            :class:`pathlib.Path`, assume a file and open for
            reading. If a dictionary or
            :class:`~collections.abc.Mapping`-like, assume the
            data contain the necessary structure and config options
            and read those in. Otherwise skip loading and use the
            :class:`configparser.ConfigParser` directly.

        Raises
        ------
        TypeError
            If ``config`` is not understood

        """
        self.hf.configure(config)
        self.rom.configure(config)
