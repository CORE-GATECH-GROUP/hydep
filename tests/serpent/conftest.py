"""Fixtures for helping test the Serpent solver

Materials and pin definitions are taken from the BEAVRS handbook,
version 2.0 - http://crpg.mit.edu/research/beavrs

Nicholas Horelik, Bryan Herman, Benoit Forget, and Kord Smith,
"Benchmark for Evaluation and Validation of Reactor Simulations,"
Proc. Int. Conf. Mathematics and Computational Methods Applied to Nuclear
Science and Engineering, Sun Valley, Idaho, May 5--9 (2013).

"""
import configparser

import pytest
import hydep
from hydep.serpent.utils import Library


@pytest.fixture(scope="module")
def beavrsMaterials():
    """Dictionary of materials from the BEAVRS specification

    Currently provides:

        fuel32 -> 3.2 wt% enriched UO2
        air -> air
        zirc4 -> zircaloy 4 cladding
        water -> unborated water
        helium -> helium
        bglass -> boroscillicate glass
        ss304 -> stainless steel 304

    """
    fuel3_2 = hydep.BurnableMaterial(
        "fuel32", mdens=10.34, temperature=900, O16=4.6026E-2, O17=1.7533E-5,
        U234=5.9959E-6, U235=7.4629E-4, U238=2.2317E-2)
    air = hydep.Material(
        "air", mdens=0.000616, O16=5.2863E-6, O17=2.0137E-9,
        N14=1.9681E-5, N15=7.1899E-8, Ar36=7.8729E-10,
        Ar38=1.4844E-10, Ar40=2.3506E-7, C12=6.7564E-9)
    zirc4 = hydep.Material(
        "zirc4", mdens=6.55, O16=3.0743E-04, O17=1.1711E-07,
        Cr50=3.2962E-06, Cr52=6.3564E-05,
        Cr53=7.2076E-06, Cr54=1.7941E-06, Fe54=8.6699E-06,
        Fe56=1.3610E-04, Fe57=3.1431E-06, Fe58=4.1829E-07,
        Zr90=2.1827E-02, Zr91=4.7600E-03, Zr92=7.2758E-03,
        Zr94=7.3734E-03, Zr96=1.1879E-03, Sn112=4.6735E-06,
        Sn114=3.1799E-06, Sn115=1.6381E-06, Sn116=7.0055E-05,
        Sn117=3.7003E-05, Sn118=1.1669E-04, Sn119=4.1387E-05,
        Sn120=1.5697E-04, Sn122=2.2308E-05, Sn124=2.7897E-05,
    )
    water = hydep.Material(
        "water", mdens=0.7405, temperature=600, B11=3.2210E-5, H1=4.9458E-2,
        H2=5.6883E-6, O16=2.4672E-2, O17=9.3981E-06)
    water.addSAlphaBeta("HinH2O")

    helium = hydep.Material(
        "helium", mdens=0.0015981, He3=3.2219E-10, He4=2.4044E-4)

    bglass = hydep.BurnableMaterial("bglass", mdens=2.26)
    for za, adens in (
        (130270, 1.74520e-03),
        (140280, 1.69250e-02),
        (140290, 8.59790e-04),
        (140300, 5.67440e-04),
        (50100, 9.65060e-04),
        (50110, 3.91890e-03),
        (80160, 4.65110e-02),
        (80170, 1.77170e-05),
    ):
        bglass[za] = adens

    ss304 = hydep.Material("ss304", mdens=8.03)
    for za, adens in (
        (140280, 9.52760e-04),
        (140290, 4.84010e-05),
        (140300, 3.19440e-05),
        (240500, 7.67780e-03),
        (240520, 1.48060e-02),
        (240530, 1.67890e-03),
        (240540, 4.17910e-04),
        (250550, 1.76040e-03),
        (260540, 3.46200e-03),
        (260560, 5.43450e-02),
        (260570, 1.23310e-03),
        (260580, 1.67030e-04),
        (280580, 5.60890e-03),
        (280600, 2.16050e-03),
        (280610, 9.39170e-05),
        (280620, 2.99460e-04),
        (280640, 7.62520e-05),
    ):
        ss304[za] = adens

    return {m.name: m for m in [ss304, bglass, fuel3_2, air, zirc4, water, helium]}


@pytest.fixture
def beavrsGuideTube(beavrsMaterials):
    """Return a guide tube"""
    return hydep.Pin(
        [0.50419, 0.5641], [beavrsMaterials[m] for m in ["water", "zirc4"]],
        outer=beavrsMaterials["water"])


@pytest.fixture
def beavrsFuelPin(beavrsMaterials):
    """Return a fuel pin using 3.2 wt% enriched fuel

    No radial divisions are performed in the fuel region

    """
    return hydep.Pin(
        [0.39218, 0.40005, 0.45720],
        [beavrsMaterials[m] for m in ["fuel32", "helium", "zirc4"]],
        outer=beavrsMaterials["water"])


@pytest.fixture
def beavrsInstrumentTube(beavrsMaterials):
    """Return an instrumentation tube"""
    return hydep.Pin(
        [0.43688, 0.48387, 0.56134, 0.60198],
        [beavrsMaterials[m] for m in ["air", "zirc4", "water", "zirc4"]],
        outer=beavrsMaterials["water"],
        name="Bare instrumentation tube")


@pytest.fixture
def beavrsControlPin(beavrsMaterials):
    """Return a control pin using boroscillicate glass poison

    No radial divisions are performed in the boroscillicate glass

    """
    return hydep.Pin(
        [0.2140, 0.2305, 0.2413, 0.4267, 0.4369, 0.4839, 0.5613, 0.6020],
        [beavrsMaterials[m] for m in ["air", "ss304", "helium", "bglass",
                                      "helium", "ss304", "water", "zirc4"]],
        outer=beavrsMaterials["water"])


@pytest.fixture
def write2x2Model(beavrsFuelPin, beavrsControlPin, beavrsGuideTube):
    template = [[0, 0], [1, 2]]
    fill = {0: beavrsFuelPin, 1: beavrsGuideTube, 2: beavrsControlPin}
    asymmetric2x2 = hydep.CartesianLattice.fromMask(1.3, template, fill)

    return hydep.Model(asymmetric2x2)


@pytest.fixture(scope="module")
def mockSerpentData(tmp_path_factory):
    files = {}
    xsdir = tmp_path_factory.mktemp("xsdata")
    for k, p in [
        [Library.ACE, "fake.xsdata"],
        [Library.DEC, "fake.dec"],
        [Library.NFY, "fake.nfy"],
    ]:
        dest = xsdir / p
        dest.touch()
        files[k] = dest

    sabf = xsdir / "acedata" / "sssth1"
    sabf.parent.mkdir()
    # Make a minimal thermal scattering file
    sabf.write_text("""lwe6.12t    0.999170  2.5507E-08   02/11/09
ENDF/B-VI.8 Data for Serpent 1.1.0 (HinH20 at 600.00K)         mat 125""")

    files[Library.SAB] = sabf
    files[Library.DATA_DIR] = xsdir

    yield files

    for p in files.values():
        if p.is_file():
            p.unlink()


@pytest.fixture
def serpentcfg(mockSerpentData):
    """Fixture with just the "hydep.serpent" configuration options"""
    options = {
        "hydep.serpent": {
            "boundary conditions": "reflective",
            "particles": 200,
            "generations per batch": 5,
            "active": 5, "skipped": 2,
            "executable": "sss2",
            "acelib": mockSerpentData[Library.ACE],
            "declib": mockSerpentData[Library.DEC],
            "nfylib": mockSerpentData[Library.NFY],
            "thermal scattering": mockSerpentData[Library.SAB],
        },
    }
    cfg = configparser.ConfigParser()
    cfg.read_dict(options)
    return cfg["hydep.serpent"]
