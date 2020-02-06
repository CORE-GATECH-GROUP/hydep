"""
Serpent writer
"""

import pathlib
import os
import warnings
from collections import deque, namedtuple
import re

import numpy

import hydep
from hydep.internal import getIsotope
from hydep.typed import TypedAttr, IterableOf
import hydep.internal.features as hdfeat

DataLibraries = namedtuple("DataLibraries", "xs decay nfy sab")


class SerpentWriter:
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

    _temps = (300, 600, 900, 1200, 1500)
    # TODO Allow config control over default material temperature
    _defaulttemp = 600
    _DEFAULT_ACELIB = "sss_endfb7u.xsdata"
    _DEFAULT_DECLIB = "sss_endfb7.dec"
    _DEFAULT_NFYLIB = "sss_endfb7.nfy"
    bcmap = {"reflective": 2, "vacuum": 1, "periodic": 3}
    model = TypedAttr("model", hydep.Model, allowNone=True)
    burnable = IterableOf("burnable", hydep.BurnableMaterial, allowNone=True)
    _groupby = 7  # Arbitrary number of gcu, mdep, fmtx arguments to write per line
    hooks = TypedAttr("hooks", hdfeat.FeatureCollection)
    _eneGridName = "energies"

    def __init__(self):
        self.model = None
        self.burnable = None
        self.base = None
        self.hooks = hdfeat.FeatureCollection()
        self.options = {}
        self.datafiles = None

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

    @staticmethod
    def commentblock(stream, msg):
        """Write a comment using the C-style multiline comments

        Parameters
        ----------
        stream : writable
            Object with a ``write`` method
        msg : str
            Single potentially multiline string to be written
            inside the comment

        """
        stream.write("/*\n * ")
        stream.write("\n * ".join(msg.split("\n")))
        stream.write("\n */\n")

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
""")

    def _writematerials(self, stream, materials):
        self.commentblock(stream, "BEGIN MATERIAL BLOCK")
        for mat in materials:
            if isinstance(mat, hydep.BurnableMaterial):
                continue
            self.writemat(stream, mat)

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
        if material.name:
            stream.write(f"% {material.name}\n")
        stream.write(
            "mat {} {}{:<9.7f}".format(
                material.id,
                "-" if material.mdens is not None else "",
                material.adens or material.mdens,
            )
        )
        if material.volume is not None:
            stream.write(f" vol {material.volume:9.7f}")
        if material.temperature is not None:
            if material.temperature not in self._temps:
                stream.write(f" tmp {material.temperature:9.7f}")
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
                libname = "_".join((table, tempkey))
                stream.write(f" moder {libname} {iso}")

        if isinstance(material, hydep.BurnableMaterial):
            stream.write(" burn 1")

        stream.write("\n")
        tlib = self._getmatlib(material)

        for isotope, adens in sorted(material.items()):
            # TODO Metastable
            stream.write(f"{isotope.z:}{isotope.a:03}.{tlib} {adens:13.9E}\n")

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

        datafiles = self._fetchDataLibraries(section)
        self.datafiles = DataLibraries(
            xs=datafiles["acelib"],
            decay=datafiles["declib"],
            nfy=datafiles["nfylib"],
            sab=datafiles["thermal scattering"],
        )

    def _fetchDataLibraries(self, section):
        datadir = section.get("data directory")
        files = {}
        missing = set()
        acekey = "acelib"
        deckey = "declib"
        nfykey = "nfylib"
        sabkey = "thermal scattering"

        for key in {acekey, deckey, nfykey}:
            v = section.get(key)
            if v is None:
                missing.add(key)
            else:
                files[key] = v

        sab = section.get(sabkey)
        if sab is None:
            missing.add(sabkey)
        else:
            files[sabkey] = pathlib.Path(sab)

        if not missing:
            return files

        if datadir is not None:
            datadir = pathlib.Path(datadir)
        else:  # try to fetch from SERPENT_DATA
            datadir = os.environ.get("SERPENT_DATA")
            if datadir is None:
                raise ValueError(
                    'Need to pass "data directory" or set SERPENT_DATA '
                    f"environment variable. Missing {missing} files"
                )
            datadir = pathlib.Path(datadir)
        if not datadir.is_dir():
            raise NotADirectoryError(datadir)

        for key, replace in {
            acekey: self._DEFAULT_ACELIB,
            deckey: self._DEFAULT_DECLIB,
            nfykey: self._DEFAULT_NFYLIB,
        }.items():
            if key in missing:
                warnings.warn("Replacing Serpent {} with {}", RuntimeWarning)
                missing.remove(key)
                files[key] = replace

        if sabkey in missing:
            # Don't check for existence, because it may not be needed
            files[sabkey] = datadir / "acedata" / "sssth1"
            missing.remove(sabkey)

        return files

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

    # TODO Replace with textwrap VVVV
    def _writeIterableOverLines(self, stream, lines, delim=" "):
        for count, line in enumerate(lines):
            stream.write(line)
            if count and count % self._groupby == 0:
                stream.write("\n")
            else:
                stream.write(delim)
        else:
            return
        if count % self._groupby != 0:
            stream.write("\n")

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
        self._writeIterableOverLines(stream, lines)
        stream.write("\n")

    def _writelocalgcu(self, stream):
        stream.write("set gcu ")
        lines = map("{}".format, (m.id for m in self.burnable))
        self._writeIterableOverLines(stream, lines)
        stream.write("\n")

    def _writeFluxDetectors(self, stream):
        self.commentblock(stream, "BEGIN FLUX DETECTORS")
        stream.write(f"det flux de {self._eneGridName}\n")
        lines = map("du {}".format, (m.id for m in self.burnable))
        self._writeIterableOverLines(stream, lines)

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
            self._writeIterableOverLines(stream, lines)

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

    def writeSteadyStateFile(self, path, timestep, power):
        """Write updated burnable materials for steady state solution

        Requires the base file with geometry, settings, and non-burnable
        materials to be written in :meth:`writeBaseFile`.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination to write the updated file
        timestep : hydep.internal.TimeStep
            Temporal information. Wil write a minor comment to the top
            of the file describing the current time step
        power : float
            Current reactor power

        Returns
        -------
        pathlib.Path
            Path of the steady-state input file

        """
        if self.burnable is None:
            raise AttributeError("No burnable material ordering set on {}".format(self))
        if self.base is None:
            raise AttributeError(
                "Base file to be included not found on {}".format(self)
            )

        steadystate = self._setupfile(path)
        with steadystate.open("w") as stream:
            self.commentblock(
                stream,
                """Steady state input file
Time step : {}
Time [d] : {:.2f}
Base file : {}""".format(
                    timestep.coarse, timestep.currentTime, self.base
                ),
            )
            stream.write('include "{}"\n'.format(self.base.absolute()))
            stream.write(f"set power {power:.7E}\n")

            for m in self.burnable:
                self.writemat(stream, m)

        return steadystate
