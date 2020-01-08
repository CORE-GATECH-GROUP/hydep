"""
Primary class for handling geometry and material information
"""

from hydep import Model, HighFidelitySolver, ReducedOrderSolver, Manager
from hydep.typed import TypedAttr
from hydep.internal import TimeStep, configmethod

class Problem(object):

    model = TypedAttr("model", Model)
    hf = TypedAttr("hf", HighFidelitySolver)
    rom = TypedAttr("rom", ReducedOrderSolver)
    dep = TypedAttr("dep", Manager)

    def __init__(self, model, hf, rom, dep):
        hf.checkCompatibility(dep)
        hf.checkCompatibility(rom)
        self.model = model
        self.hf = hf
        self.rom = rom
        self.dep = dep

    def beforeMain(self):
        self.dep.beforeMain(self.model)
        self.hf.beforeMain(self.model, self.dep.burnable)
        self.rom.beforeMain(self.model, self.dep.burnable)

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
