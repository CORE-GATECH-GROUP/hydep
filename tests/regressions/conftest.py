import pytest
import hydep

from tests.regressions import ProblemProxy

@pytest.fixture
def toy2x2Problem(simpleChain, toy2x2lattice):
    # Include the chain so reactions are present
    model = hydep.Model(toy2x2lattice)
    model.differentiateBurnableMaterials(updateVolumes=False)

    burnable = tuple(model.root.findBurnableMaterials())
    for m in burnable:
        m.volume = 1.0

    yield ProblemProxy(model, burnable)


