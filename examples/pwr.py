import numpy
import hydep

# Build materials

fuel = hydep.BurnableMaterial("fuel", mdens=10.4, U235=8.0e-4, U238=2.5e-2, O16=4.6e-4)
clad = hydep.Material("clad", mdens=6.5, Zr90=4.3E-2)
water = hydep.Material("water", mdens=1.0, H1=4.7e-2, O16=2.4e-2)

# Build geometry

fuelPin = hydep.Pin([0.4005, 0.42], [fuel, clad], water, name="fuel")
guidePin = hydep.Pin([0.5, 0.55], [water, clad], water, name="guide")

lattice = hydep.CartesianLattice(3, 3, 1.26)
# TODO Fix numpy.asarray for nested pin lists
pins = numpy.empty((3, 3), dtype=object)
pins.fill(fuelPin)
pins[1, 1] = guidePin

lattice.array = pins

print(lattice.countBurnableMaterials())
