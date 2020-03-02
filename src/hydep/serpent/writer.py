"""
Serpent writer
"""

import pathlib
import warnings
import re
from textwrap import TextWrapper

import numpy

import hydep
from hydep.constants import SECONDS_PER_DAY
from hydep.internal import getIsotope
from hydep.typed import TypedAttr, IterableOf
import hydep.internal.features as hdfeat

from .utils import findLibraries, findProblemIsotopes, ProblematicIsotopes


class BaseWriter:
    """Parent class for writing basic information

    Parameters
    ----------
    model : hydep.Model, optional
        Initial model

    Attributes
    ----------
    model : hydep.Model or None
        Entry point to geometry and materials. Overwriting this is strongly
        discouraged.

    """

    _temps = (300, 600, 900, 1200, 1500)
    # TODO Allow config control over default material temperature
    _defaulttemp = 600
    _DEFAULT_ACELIB = "sss_endfb7u.xsdata"
    _DEFAULT_DECLIB = "sss_endfb7.dec"
    _DEFAULT_NFYLIB = "sss_endfb7.nfy"
    bcmap = {"reflective": 2, "vacuum": 1, "periodic": 3}
    hooks = TypedAttr("hooks", hdfeat.FeatureCollection)
    burnable = IterableOf("burnable", hydep.BurnableMaterial, allowNone=True)

    _eneGridName = "energies"

    def __init__(self):
        self._model = None
        self.burnable = None
        self.hooks = hdfeat.FeatureCollection()
        self.options = {}
        self.datafiles = None
        self._buleads = {}
        self._problemIsotopes = ProblematicIsotopes(missing=set(), replacements={})
        self._textwrapper = TextWrapper(width=75)
        self._commenter = TextWrapper(
            width=75, initial_indent=" * ", subsequent_indent=" * ",
        )

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        if m is None:
            self._model = None
        elif not isinstance(m, hydep.Model):
            raise TypeError(f"model must be {hydep.Model}, not {type(m)}")
        elif self._model is not None:
            raise AttributeError(f"Refusing to overwrite model on {self}")
        self._model = m

    @staticmethod
    def _setupfile(path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        if path.is_file():
            return path
        if path.exists():
            raise IOError("{} exists and is not a file.".format(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def commentblock(self, stream, msg):
        """Write a comment using the C-style multiline comments

        Parameters
        ----------
        stream : writable
            Object with a ``write`` method
        msg : str
            Single potentially multiline string to be written
            inside the comment

        """
        stream.write("/*\n")
        stream.write(self._commenter.fill(msg))
        stream.write("\n */\n")

    def _findSABTables(self, materials, sab: pathlib.Path) -> dict:
        replace = {
            "HinH2O": "HinH20",
            "DinH2O": "DinH20",
        }  # Error in early distributed sssth1 files

        found = set()
        for mat in materials:
            for table in mat.sab:
                found.add((table, f"{mat.temperature:.2f}"))

        if not found:
            return {}

        if not sab.is_file():
            raise FileNotFoundError(
                f"Model contains S(a,b) tables, but file {sab} not found"
            )

        patterns = {}
        for table, temp in found:
            tail = r"\({} at {}K\)".format(
                replace.get(table, table), temp.replace(".", r"\."),
            )
            patterns[table, temp] = re.compile(tail)

        tables = {}

        with sab.open("r") as stream:
            prev = None
            for line in stream:
                for key, pattern in patterns.items():
                    if pattern.search(line) is not None:
                        tables[key] = prev.split()[0]
                        break
                else:
                    prev = line
                    continue
                del patterns[key]
                if patterns:
                    prev = line
                    continue
                else:
                    break

        return tables

    def updateProblemIsotopes(self, zais, xsfile=None):
        """Search through an ACE file and update special isotopes

        Some isotopes may be included in the depletion chain, but
        not in the Serpent library. Metastable isotopes may be
        found under a different ZA number. Am242_m1 can sometimes
        be found at 95342 rather than 95242.

        Parameters
        ----------
        zais : iterable of [int, int, int]
            Z, A, I triplets of isotopes that could be included in
            future transport simulations.
        xsfile : str or pathlib.Path, optional
            Path to cross section look up table, usually ending in
            ``.xsdata``. If not provided, then :attr:`datafiles`
            must be configured via :meth:`configure`

        Returns
        -------
        ProblematicIsotopes
            Isotopes that are requested but are either missing from
            the library, or found under a different ZA number

        """
        if xsfile is None:
            if self.datafiles is None:
                raise ValueError(
                    "Either provide xsfile directly, or include when " "configuring"
                )
            xsfile = self.datafiles.xs

        with xsfile.open("r") as s:
            p = findProblemIsotopes(s, zais)

        self._problemIsotopes.missing.update(p.missing)
        self._problemIsotopes.replacements.update(p.replacements)

        return p

    def _writesettings(self, stream, sabLibraries):
        self.commentblock(stream, "BEGIN SETTINGS BLOCK")

        libraries = []
        for attr, lib in (("xs", "ace"), ("decay", "dec"), ("nfy", "nfy")):
            userlib = getattr(self.datafiles, attr)
            libraries.append(f'set {lib}lib "{userlib}"')
        stream.write("\n".join(libraries) + "\n")

        stream.write("set pop {} ".format(self.options["particles"]))

        gen = self.options["generations per batch"]
        active = self.options["active"]
        skipped = self.options["skipped"]
        stream.write("{} {}".format(gen * active, gen * skipped))
        # Batching is problematic for power reconstruction through INF_FLX
        # https://ttuki.vtt.fi/serpent/viewtopic.php?f=25&t=3306
        stream.write(" {:.5f} % {}\n".format(self.options.get("k0", 1.0), gen))

        stream.write("set bc")
        for value in self.options.get("bc", [1]):
            stream.write(f" {value}")
        stream.write("\n")

        if sabLibraries:
            sab = []
            for (tablename, tempstr), tablelib in sabLibraries.items():
                sab.append(f"therm {tablename}_{tempstr} {tablelib}")
            stream.write("\n".join(sab) + "\n")

        seed = self.options.get("seed")
        if seed is not None:
            stream.write(f"set seed {seed}\n")

        stream.write(
            f"""% Hard set one group [0, 20] MeV for all data
ene {self._eneGridName} 2 1 0 20
set nfg {self._eneGridName}
"""
        )

    def _getMaterialOptions(self, material):
        if material.adens is None:
            density = f"-{material.mdens:<9.7f}"
        else:
            density = f"{material.adens:<9.7f}"

        args = [f"mat {material.id} {density}"]

        if material.volume is not None:
            args.append(f"vol {material.volume:9.7f}")
        if material.temperature is not None:
            if material.temperature not in self._temps:
                args.append(f"tmp {material.temperature:9.7f}")
            tempkey = f"{material.temperature:.2f}"
            for table in material.sab:
                if table.startswith("Graphite"):
                    iso = "12000"
                elif table.startswith("H"):
                    iso = "1001"
                elif table.startswith("D"):
                    iso = "1002"
                else:
                    raise ValueError(f"Unknown S(a,b) table {table}")
                args.append(f"moder {table}_{tempkey} {iso}")

        if isinstance(material, hydep.BurnableMaterial):
            args.append("burn 1")

        return args

    def writemat(self, stream, material):
        """Write a valid Serpent material input

        Parameters
        ----------
        stream : writable
            Object with a ``write`` method to give the material
            definition
        material : hydep.Material
            Some material to write

        """
        start = [f"% {material.name}"] if material.name else []
        args = self._getMaterialOptions(material)
        start.append(" ".join(args))

        stream.write("\n".join(start) + "\n")

        pairs = ((iso.triplet, adens) for iso, adens in sorted(material.items()))
        # TODO Configurable threshold
        self.writeMatIsoDef(stream, pairs, self._getmatlib(material))
        stream.write("\n")

    def writeMatIsoDef(self, stream, pairs, tlib, threshold=1e-20) -> float:
        """Write ZAI, atom density pairs, considering problem isotopes

        Parameters
        ----------
        stream : writeable
            Destination of the isotope block
        pairs : iterable of ((int, int, int), float)
            Isotope ZAI and corresponding atom density [#/b/cm]
        tlib : str
            Temperature-specific cross section library, e.g. ``"09c"``
            for 900K continuous energy cross sections
        threshold : float, optional
            Threshold for writing atom densities. Any isotope with
            density under this value will not be written

        Returns
        -------
        float
            Sum of atom density for missing isotopes not written

        See Also
        --------
        * :meth:`updateProblemIsotopes` - which isotopes don't exist
          in the current library, or exist under different names
        * :meth:`configure` - Configuration method that sets the
          cross section library used

        """
        missing = 0
        lines = []
        for (z, a, i), adens in pairs:
            if adens < threshold:
                missing += adens
                continue
            # Check missing
            if (z, a, i) in self._problemIsotopes.missing:
                missing += adens
                continue
            # Get Z, A for isotope that may be listed under a different
            # name, e.g. metastable isotopes
            z, a = self._problemIsotopes.replacements.get((z, a, i), (z, a))
            lines.append(f"{z}{a:03}.{tlib} {adens:13.9E}")
        stream.write("\n".join(lines))
        return missing

    def _getmatlib(self, mat):
        """Return the continuous energy library, "03c", given material"""
        if mat.temperature is not None:
            if mat.temperature < min(self._temps):
                warnings.warn(
                    "Temperature {:.5f} for {} too low. Using {}".format(
                        mat.temperature, repr(mat), self._defaulttemp
                    )
                )
                temp = self._defaulttemp
            else:
                temp = mat.temperature
        else:
            temp = self._defaulttemp

        for ix, t in enumerate(self._temps, start=1):
            if t == temp:
                break
            elif t > temp:
                t = self._temps[ix - 1]
                break
        return "{:02}c".format(t // 100)

    def _writematerials(self, stream, materials):
        self.commentblock(stream, "BEGIN MATERIAL BLOCK")
        for mat in materials:
            self.writemat(stream, mat)

    def _writegeometry(self, stream):
        self.commentblock(stream, "BEGIN GEOMETRY BLOCK")
        rootid = self.writeUniverse(stream, self.model.root, {})
        # TODO Pull bounds from model if given
        self._writecellbounds(stream, self.model.root, rootid, 0)

    def writeUniverse(self, stream, u, memo):
        """Write the geometry definition for this material

        Parameters
        ----------
        stream : writable
            Object onto which to write the geometry
        u : hydep.lib.Universe
            Universe to be written
        memo : dict[str, str]
            Dictionary mapping previously written universe ids to their
            Serpent universe number

        Raises
        ------
        TypeError
            If the type is not :class:`hydep.Pin`,
            :class:`hydep.CartesianLattice`, or
            :class:`hydep.LatticeStack`

        """
        if isinstance(u, hydep.Pin):
            return self._writepin(stream, u, memo)
        if isinstance(u, hydep.CartesianLattice):
            return self._writelattice(stream, u, memo)
        if isinstance(u, hydep.LatticeStack):
            return self._writestack(stream, u, memo)
        if isinstance(u, hydep.InfiniteMaterial):
            return self._writeInfMaterial(stream, u, memo)
        raise TypeError(type(u))

    def _writepin(self, stream, pin, memo):
        previous = memo.get(pin.id)
        if previous is not None:
            return previous

        writeas = f"p{pin.id}"
        memo[pin.id] = writeas

        if pin.name is not None:
            stream.write(f"% {pin.name}\n")

        infiniteCells = []
        pindef = [f"pin {writeas}"]

        # Drop universes inside pin definition like magic
        # https://ttuki.vtt.fi/serpent/viewtopic.php?f=18&t=3092#p9345

        for ix, (r, m) in enumerate(zip(pin.radii, pin.materials)):
            if isinstance(m, hydep.BurnableMaterial):
                surf = f"{pin.id}_r{ix}"
                infiniteCells.append(
                    f"surf {surf} inf\ncell {surf} {m.id} {m.id} -{surf}"
                )
                pindef.append(f"fill {m.id} {r:.7f}")
            else:
                pindef.append(f"{m.id} {r:.7f}")
        pindef.append(f"{pin.outer.id}\n\n")  # extra new line after pin

        stream.write("\n".join(infiniteCells + pindef))

        return writeas

    def _writelattice(self, stream, lat, memo):
        previous = memo.get(lat.id)
        if previous is not None:
            return previous

        outermost = f"cl{lat.id}"
        if lat.outer is None:
            innermost = outermost
        else:
            innermost = outermost + "_0"

        memo[lat.id] = outermost

        univrows = []

        for row in lat.array:
            items = []
            for univ in row:
                uid = memo.get(univ.id)
                if uid is None:
                    uid = self.writeUniverse(stream, univ, memo)
                    memo[univ.id] = uid
                items.append(uid)
            univrows.append(items)

        if lat.name is not None:
            stream.write(f"% {lat.name}\n")

        stream.write(f"lat {innermost} 1 0.0 0.0 {lat.nx} {lat.ny} {lat.pitch:.5f}\n")
        while univrows:
            stream.write(" ".join(map(str, univrows.pop())) + "\n")

        if lat.outer is None:
            return outermost

        self._writecellbounds(stream, lat, innermost, outermost, lat.outer.id)

        return outermost

    @staticmethod
    def _writecellbounds(stream, universe, filler, universenumber, outer="outside"):
        bounds = universe.bounds
        xybounds = " ".join(
            map("{:.5f}".format, (bounds.x[0], bounds.x[1], bounds.y[0], bounds.y[1]),)
        )
        if bounds.z is None or (-bounds.z[0] == bounds.z[1] == numpy.inf):
            surf = f"rect {xybounds}"
        else:
            surf = f"cuboid {xybounds} {bounds.z[0]:.5f} {bounds.z[1]:.5f}"
        stream.write(
            f"""
surf {universe.id}_x {surf}
cell {universe.id}_1 {universenumber} fill {filler} -{universe.id}_x
cell {universe.id}_2 {universenumber} {outer} {universe.id}_x
"""
        )

    def _writestack(self, stream, lstack, memo):
        previous = memo.get(lstack.id)
        if previous is not None:
            return previous

        writeas = f"ls{lstack.id}"

        memo[lstack.id] = writeas

        subids = []
        for item in lstack:
            uid = memo.get(item.id)
            if uid is None:
                uid = self.writeUniverse(stream, item, memo)
                memo[item.id] = uid
            subids.append(uid)

        if lstack.name is not None:
            stream.write(f"% {lstack.name}\n")

        stream.write(f"lat {writeas} 9 0.0 0.0 {lstack.nLayers}\n")
        for lower, sub in zip(lstack.heights[:-1], subids):
            stream.write(f"{lower:.5f} {sub}\n")

        return writeas

    def _writeInfMaterial(self, stream, infmat, memo):
        previous = memo.get(infmat.id)
        if previous is not None:
            return previous

        writeas = f"inf{infmat.id}"
        memo[infmat.id] = writeas

        if infmat.material.name is not None:
            stream.write(f"% Infinite region filled with {infmat.material.name}\n")

        stream.write(
            f"""surf {writeas} inf
cell {writeas} {writeas} {infmat.material.id} -{writeas}
"""
        )

        return writeas

    def configure(self, section, level):
        """Configure the writer

        Parameters
        ----------
        section : configparser.SectionProxy
            This specific set of configuration options
        level : int
            Depth or specificity. Currently support three levels:
            ``hydep``, ``hydep.montecarlo``, and ``hydep.serpent``.
            Options like verbosity and initial :math:`k` will be
            read read from level ``0``, with more specifics in levels
            ``1`` and ``2``

        """
        k0 = section.getfloat("initial k")
        if k0 is not None:
            self.options["k0"] = k0

        bc = section.get("boundary conditions")
        if bc is not None:
            self._parseboundaryconditions(bc.replace(",", " ").split())

        if level == 0:
            return

        # Just Monte Carlo settings

        seed = section.getint("random seed")
        if seed is not None:
            self.options["seed"] = seed

        for key in [
            "particles",
            "generations per batch",
            "active",
            "skipped",
        ]:
            value = section.getint(key)
            if value is not None:
                self.options[key] = value

        if level == 1:
            return

        # Serpent settings

        files = [
            pathlib.Path(p)
            for p in [
                section.get("acelib", self._DEFAULT_ACELIB),
                section.get("declib", self._DEFAULT_DECLIB),
                section.get("nfylib", self._DEFAULT_NFYLIB),
            ]
        ]

        sab = section.get("thermal scattering")
        if sab is not None:
            sab = pathlib.Path(sab)

        datadir = section.get("data directory")
        if datadir is not None:
            datadir = pathlib.Path(datadir)

        self.datafiles = findLibraries(*files, sab, datadir)

    def _parseboundaryconditions(self, conds):
        bc = []
        if len(conds) == 1:
            conds = conds * 3
        for c in conds:
            bcval = self.bcmap.get(c)
            if bcval is None:
                raise ValueError(
                    "Unsure how to process boundary condition {}. "
                    "Supported values are {}".format(c, ", ".join(self.bcmap))
                )
            bc.append(bcval)
        self.options["bc"] = bc

    def _writehooks(self, stream):
        self.commentblock(stream, "BEGIN HOOKS")
        if hdfeat.FISSION_MATRIX in self.hooks:
            self._writefmtx(stream)
        if hdfeat.HOMOG_LOCAL in self.hooks:
            self._writelocalgcu(stream)
        else:
            self._writeFluxDetectors(stream)
        if hdfeat.MICRO_REACTION_XS in self.hooks:
            self._writelocalmicroxs(stream)

    def _writefmtx(self, stream):
        stream.write("set fmtx 2 ")
        lines = map("{}".format, (m.id for m in self.burnable))
        stream.write(self._textwrapper.fill("\n".join(lines)) + "\n")

    def _writelocalgcu(self, stream):
        stream.write("set gcu ")
        lines = map("{}".format, (m.id for m in self.burnable))
        stream.write(self._textwrapper.fill("\n".join(lines)) + "\n")

    def _writeFluxDetectors(self, stream):
        self.commentblock(stream, "BEGIN FLUX DETECTORS")
        stream.write(f"det flux de {self._eneGridName}\n")
        lines = map("du {}".format, (m.id for m in self.burnable))
        stream.write(self._textwrapper.fill("\n".join(lines)) + "\n")

    def _writelocalmicroxs(self, stream):
        self.commentblock(
            stream,
            """BEGIN MICROSCOPIC REACTION XS BLOCK
Need to trick Serpent into given this information, but we don't want a ton
of depletion. Add a single one day step here. Maybe hack something later""",
        )
        stream.write("dep daystep 1\nset pcc 0\n")
        for m in self.burnable:
            stream.write(f"set mdep {m.id} 1.0 1 {m.id}\n")
            reactions = self._getReactions(set(m))
            lines = (f"{z} {m}" for z, m in sorted(reactions))
            stream.write(self._textwrapper.fill("\n".join(lines)) + "\n")

    @staticmethod
    def _getReactions(isotopes):
        reactions = set()
        previous = {None}
        while isotopes:
            iso = isotopes.pop()
            if iso in previous:
                continue
            previous.add(iso)
            if iso.zai > 992550:  # Serpent upper limit
                continue
            for reaction in iso.reactions:
                if reaction.target not in previous:
                    isotopes.add(reaction.target)
                reactions.add((iso.zai, reaction.mt))
            for decay in iso.decayModes:
                if decay.target not in previous:
                    isotopes.add(decay.target)
            if iso.fissionYields is None:
                continue
            for prod in (getIsotope(zai=z) for z in iso.fissionYields.products):
                if prod not in previous:
                    isotopes.add(prod)
        return reactions


class SerpentWriter(BaseWriter):
    """Class responsible for writing Serpent input files

    Must be properly configured with particle history
    information using :meth:`configure`. Model information
    and burnable materials must be provided to
    :attr:`model` and :attr:`burnable` prior to calling
    :meth:`writeBaseFile` and :meth:`writeSteadyStateFile`

    Attributes
    ----------
    model : hydep.Model or None
        Geometry and material information. Must be provided
    burnable : Iterable[hydep.BurnableMaterial] or None
        Ordered iterable of burnable materials. Must be provided.
    basefile : pathlib.Path or None
        The primary input file that contians all geometry, settings,
        and non-burnable material definitions
    hooks : hydep.internal.features.FeatureCollection
        Each entry indicates a specific type of physics that
        must be run.
    options : dict
        Dictionary of various attributes to create the base file
    datafiles : None or DataLibraries
        Configured through :meth:`configure`

    """

    def __init__(self):
        super().__init__()
        self.base = None

    def _writematerials(self, stream, materials):
        self.commentblock(stream, "BEGIN MATERIAL BLOCK")
        for mat in materials:
            if isinstance(mat, hydep.BurnableMaterial):
                continue
            self.writemat(stream, mat)

    def writeBaseFile(self, path):
        """Write the base input file to be included later

        Parameters
        ----------
        path : str or pathlib.Path
            Path of file to be written. If it is an existing file
            it will be overwritten

        Raises
        ------
        IOError
            If the path indicated exists and is not a file
        AttributeError
            If :attr:`model` nor :attr:`options` have been
            properly set

        """
        if self.model is None:
            raise AttributeError("Geometry not passed to {}".format(self))
        if not self.options or self.datafiles is None:
            raise AttributeError("Not well configured")

        self.base = self._setupfile(path)

        materials = tuple(self.model.root.findMaterials())
        sabLibraries = self._findSABTables(materials, self.datafiles.sab)

        with self.base.open("w") as stream:
            self._writematerials(stream, materials)
            self._writegeometry(stream)
            self._writesettings(stream, sabLibraries)
            if self.hooks:
                self._writehooks(stream)

    def writeSteadyStateFile(self, path, compositions, timestep, power, final=False):
        """Write updated burnable materials for steady state solution

        Requires the base file with geometry, settings, and non-burnable
        materials to be written in :meth:`writeBaseFile`.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination to write the updated file
        compositions : hydep.internal.CompBundle
            Current burnable material compositions for this time step
        timestep : hydep.internal.TimeStep
            Temporal information. Will write a minor comment to the top
            of the file describing the current time step and time [d],
            assuming ``timestep.currentTime`` is in seconds.
        power : float
            Current reactor power [W]
        final : bool, optional
            If ``True``, no depletion information will be written.

        Returns
        -------
        pathlib.Path
            Path of the steady-state input file, resolved to be
            absolute path.

        """
        if self.burnable is None:
            raise AttributeError(f"No burnable material ordering set on {self}")
        if self.base is None:
            raise AttributeError(f"Base file to be included not found on {self}")

        steadystate = self._setupfile(path)
        with steadystate.open("w") as stream:
            stream.write(f"""/*
 * Steady state input file
 * Time step : {timestep.coarse}
 * Time [d] : {timestep.currentTime/SECONDS_PER_DAY:.2f}
 * Base file : {self.base}
 */
include "{self.base.resolve()}"
set power {power:.7E}\n""")

            zais = tuple((iso.triplet for iso in compositions.isotopes))

            for ix, densities in enumerate(compositions.densities):
                matprops = self._buleads.get(ix)
                if matprops is None:
                    try:
                        mat = self.burnable[ix]
                    except IndexError as ie:
                        raise ie from KeyError(f"Cannot find burnable material {ix}")
                    matdef = " ".join(self._getMaterialOptions(mat))
                    tlib = self._getmatlib(mat)
                    self._buleads[ix] = matdef, tlib
                else:
                    matdef, tlib = matprops

                if final:
                    # TODO Turn off all hooks except flux for final step
                    # TODO Only load decay, nfy for non-final steps
                    # META do we need decay, nfy libraries at all?
                    matdef = matdef.replace(" burn 1", "")

                stream.write(f"{matdef}\n")
                self.writeMatIsoDef(stream, zip(zais, densities), tlib)
                stream.write("\n")

        return steadystate
