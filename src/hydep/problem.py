"""
Primary class for handling geometry and material information
"""

from hydep import Model, HighFidelitySolver, ReducedOrderSolver, Manager
from hydep.typed import TypedAttr
from hydep.internal import TimeStep

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
        self.beforeMain()
        result = None
        # Hooks?
