"""
Serpent writer
"""

import os
import pathlib
import warnings
import re
from textwrap import TextWrapper
import struct
from collections import OrderedDict
from collections.abc import Sequence

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
        discouraged. Required before :meth:`writeMainFile`
    burnable : sequence of hydep.BurnableMaterial or None
        Ordering of burnable materials. Required before :meth:`writeMainFile`

    """

    _temps = (300, 600, 900, 1200, 1500)
    # TODO Allow config control over default material temperature
    _defaulttemp = 600
    hooks = TypedAttr("hooks", hdfeat.FeatureCollection)
    burnable = IterableOf("burnable", hydep.BurnableMaterial, allowNone=True)

    _eneGridName = "energies"

    def __init__(self):
        self._model = None
        self.burnable = None
        self.hooks = hdfeat.FeatureCollection()
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

    def updateProblemIsotopes(self, zais, xsfile):
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
        xsfile : str or pathlib.Path
            Path to cross section look up table, usually ending in
            ``.xsdata``.

        Returns
        -------
        ProblematicIsotopes
            Isotopes that are requested but are either missing from
            the library, or found under a different ZA number

        """
        with xsfile.open("r") as s:
            p = findProblemIsotopes(s, zais)

        self._problemIsotopes.missing.update(p.missing)
        self._problemIsotopes.replacements.update(p.replacements)

        return p

    def _writesettings(self, stream, sabLibraries, settings):
        self.commentblock(stream, "BEGIN SETTINGS BLOCK")

        serpentSettings = settings.serpent

        stream.write(f"""set acelib "{serpentSettings.acelib}"
set declib "{serpentSettings.declib}"
set nfylib "{serpentSettings.nfylib}"
""")
        particles = serpentSettings.particles
        active = serpentSettings.active
        inactive = serpentSettings.inactive
        gen = serpentSettings.generationsPerBatch

        if any([x is None for x in [particles, active, inactive, gen]]):
            raise ValueError("Particle settings not well configured")
        k0 = serpentSettings.k0 or 1.0

        # Batching is problematic for power reconstruction through INF_FLX
        # https://ttuki.vtt.fi/serpent/viewtopic.php?f=25&t=3306
        stream.write(
            f"set pop {particles} {gen * active} {inactive * gen} "
            f"{k0:.5f} % {gen}\n"
        )

        bc = self._parseboundaryconditions(settings.boundaryConditions)
        stream.write(f"set bc {' '.join(bc)}\n")

        if sabLibraries:
            for (tablename, tempstr), tablelib in sabLibraries.items():
                stream.write(f"therm {tablename}_{tempstr} {tablelib}\n")

        if serpentSettings.seed is not None:
            stream.write(f"set seed {serpentSettings.seed}\n")

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

        bounds = self.model.bounds
        if bounds is None:
            bounds = self.model.root.bounds

        self._writeCellAndBoundSurf(
            stream,
            self.model.root.id,
            0,
            rootid,
            bounds,
            outer="outside",
        )

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

        stream.write(
            f"lat {innermost} 1 {lat.center[0]:.3f} {lat.center[1]:.3f} "
            f"{lat.nx} {lat.ny} {lat.pitch:.5f}\n"
        )
        while univrows:
            stream.write(" ".join(map(str, univrows.pop())) + "\n")

        if lat.outer is not None:
            self._writeCellAndBoundSurf(
                stream,
                lat.id,
                outermost,
                innermost,
                lat.outer.id,
                lat.bounds,
            )

        return outermost

    @staticmethod
    def _writeCellAndBoundSurf(
        stream,
        universeID,
        universeNumber,
        filler,
        bounds,
        outer="outside",
    ):
        xybounds = " ".join(
            map("{:.5f}".format, (bounds.x[0], bounds.x[1], bounds.y[0], bounds.y[1]),)
        )
        if bounds.z is None or (-bounds.z[0] == bounds.z[1] == numpy.inf):
            surf = f"rect {xybounds}"
        else:
            surf = f"cuboid {xybounds} {bounds.z[0]:.5f} {bounds.z[1]:.5f}"
        stream.write(
            f"""
surf {universeID}_x {surf}
cell {universeID}_1 {universeNumber} fill {filler} -{universeID}_x
cell {universeID}_2 {universeNumber} {outer} {universeID}_x
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

        # TODO Support for outer material

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

    def _parseboundaryconditions(self, conds):
        bc = []
        bcmap = {"reflective": "2", "vacuum": "1", "periodic": "3"}
        for c in conds:
            bcval = bcmap.get(c)
            if bcval is None:
                raise ValueError(
                    "Unsure how to process boundary condition {}. "
                    "Supported values are {}".format(c, ", ".join(bcmap))
                )
            bc.append(bcval)
        return bc

    def _writehooks(self, stream):
        self.commentblock(stream, "BEGIN HOOKS")
        self._writeFluxDetectors(stream)
        if hdfeat.FISSION_MATRIX in self.hooks:
            self._writefmtx(stream)
        if hdfeat.HOMOG_LOCAL in self.hooks:
            self._writelocalgcu(stream)
        if hdfeat.MICRO_REACTION_XS in self.hooks:
            self._writeMdep(stream)

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

    def _writeMdep(self, stream):
        for m in self.burnable:
            stream.write(f"set mdep {m.id} 1.0 1 {m.id}\n")
            reactions = self._getReactions(set(m))
            lines = (f"{z} {m}" for z, m in sorted(reactions))
            stream.write(self._textwrapper.fill("\n".join(lines)) + "\n")

    @staticmethod
    def _getReactions(isotopes):
        # TODO Cache the reactions given a set of isotopes
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

    def writeMainFile(self, path, settings):
        """Write the main input file

        Parameters
        ----------
        path : str or pathlib.Path
            Path of file to be written. If it is an existing file
            it will be overwritten
        settings : hydep.settings.Settings
            Various configuration settings

        Returns
        -------
        pathlib.Path
            Absolute path to the file that has been written

        Raises
        ------
        hydep.GeometryError
            If :attr:`model` is unbounded
        OSError
            If the path indicated exists and is not a file

        """
        if self.model is None:
            raise AttributeError(f"Geometry not passed to {self}")
        if not self.model.isBounded():
            raise hydep.GeometryError(
                f"Model is unbounded with boundaries {self.model.bounds}"
            )
        if self.burnable is None or not len(self.burnable):
            raise AttributeError(f"No burnable materials found on {self}")

        path = self._setupfile(path)

        # Resolve data libraries
        files = []
        ace = settings.serpent.acelib
        if ace is None:
            raise AttributeError("Cross section library <acelib> not configured")
        files.append(ace)
        dec = settings.serpent.declib
        if dec is None:
            raise AttributeError("Cross section library <declib> not configured")
        files.append(dec)
        nfy = settings.serpent.nfylib
        if nfy is None:
            raise AttributeError("Cross section library <nfylib> not configured")
        files.append(nfy)

        datafiles = findLibraries(*files, settings.serpent.sab, settings.serpent.datadir)

        materials = tuple(self.model.root.findMaterials())
        sabLibraries = self._findSABTables(materials, datafiles.sab)

        with path.open("w") as stream:
            self._writematerials(stream, materials)
            self._writegeometry(stream)
            self._writesettings(stream, sabLibraries, settings)
            if self.hooks:
                self._writehooks(stream)

        return path


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

    """

    def __init__(self):
        super().__init__()
        self.base = None

    def writeBaseFile(self, path, settings):
        """Write the main input file

        The path is stored to be included later in
        :meth:`writeSteadyStateFile`. Burnable materials are
        not written, as they are also written in
        :meth:`writeSteadyStateFile`

        Parameters
        ----------
        path : str or pathlib.Path
            Path of file to be written. If it is an existing file
            it will be overwritten

        Raises
        ------
        IOError
            If the path indicated exists and is not a file

        Returns
        -------
        pathlib.Path
            Absolute path to the file that has been written

        """
        base = self.writeMainFile(path, settings)
        self.base = base
        return base

    def _writematerials(self, stream, materials):
        self.commentblock(stream, "BEGIN MATERIAL BLOCK")
        for mat in materials:
            if isinstance(mat, hydep.BurnableMaterial):
                continue
            self.writemat(stream, mat)

    def _writeMdep(self, stream):
        self.commentblock(
            stream,
            """BEGIN MICROSCOPIC REACTION XS BLOCK
Need to trick Serpent into given this information, but we don't want a ton
of depletion. Add a single one day step here. Maybe hack something later""",
        )
        stream.write("dep daystep 1\nset pcc 0\n")
        super()._writeMdep(stream)

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
            stream.write(
                f"""/*
 * Steady state input file
 * Time step : {timestep.coarse}
 * Time [d] : {timestep.currentTime/SECONDS_PER_DAY:.2f}
 * Base file : {self.base}
 */
include "{self.base.resolve()}"
set power {power:.7E}\n"""
            )

            zais = tuple((iso.triplet for iso in compositions.isotopes))

            for ix, densities in enumerate(compositions.densities):
                matprops = self._buleads.get(ix)
                if matprops is None:
                    try:
                        mat = self.burnable[ix]
                    except IndexError as ie:
                        raise KeyError(f"Cannot find burnable material {ix}") from ie
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


class ExtDepWriter(BaseWriter):
    """Writer reponsible for setting up the external depletion

    Relies on a patched version of Serpent that allows Serpent to
    read in new compositions at new depletion intervals. Only supports
    the signal-based communication now.
    """

    _FAKE_BURNUP = 12345.0

    def __init__(self):
        super().__init__()
        self._burnable = None
        self.compFile = None
        self._names = None
        self._allowedZAI = None

    @property
    def burnable(self):
        return self._burnable

    @burnable.setter
    def burnable(self, mats):
        if mats is None:
            self._burnable = None
            self._names = None
            return
        if not isinstance(mats, Sequence):
            raise TypeError(
                f"burnable must be Sequence of burnable material, not {type(mats)}"
            )

        names = OrderedDict()
        for item in mats:
            if not isinstance(item, hydep.BurnableMaterial):
                raise TypeError(
                    f"burnable must be Sequence of burnable material, found {type(item)}"
                )
            names[str(item.id).encode()] = {"adens": item.adens, "mdens": item.mdens}

        self._burnable = mats
        self._names = names

    def writeCouplingFile(self, path, timesteps, powers, settings):
        base = self.writeMainFile(path, settings)

        if self.compFile is None:
            self.compFile = base.with_suffix(".exdep")

        # Write additions
        with base.open("a") as stream:
            stream.write(
                f"""
/*
 * BEGIN EXTERNAL DEPLETION INTERFACE
 *
 * Relies on a patched version of Serpent that supports the
 * extdep setting
 */
set pcc ce
set extdep 1 "{self.compFile}"
set ppid {os.getpid()}
"""
            )
            for sec, powr in zip(timesteps, powers):
                stream.write(
                    f"dep daystep {sec / SECONDS_PER_DAY:.3E} set power {powr}\n"
                )
        return base

    def _readZAI(self):
        present = set()
        # Read through restart file and find loaded isotopes from
        # first material
        with self.compFile.open("rb") as stream:
            buf = stream.read(struct.calcsize("l"))

            (namelen,) = struct.unpack("l", buf)
            assert namelen > 0, (self.compFile, namelen)

            (name,) = struct.unpack(
                f"{namelen}s", stream.read(struct.calcsize(f"{namelen}s"))
            )
            assert name in self._names, (self.compFile, name.decode())

            # Skip days, nominal burnup
            stream.read(struct.calcsize("2d"))

            (nnucs,) = struct.unpack("l", stream.read(struct.calcsize("l")))
            assert nnucs > 0, nnucs

            # Skip adens, mdens, material burnup
            stream.read(struct.calcsize("3d"))

            buf = stream.read(nnucs * struct.calcsize("ld"))

            for z, _a in struct.iter_unpack("ld", buf):
                if z > 0:
                    present.add(z)

        return present

    def updateFromRestart(self):
        """Fetch Serpent adens, mdens from file"""
        assert self._names is not None

        zais = set()

        with self.compFile.open("rb") as stream:
            buf = stream.read(struct.calcsize("l"))
            assert buf

            while buf:
                (namelen,) = struct.unpack("l", buf)
                assert namelen > 0, namelen

                (bname,) = struct.unpack(
                    f"{namelen}s", stream.read(struct.calcsize(f"{namelen}s")),
                )
                mdata = self._names.get(bname)
                assert mdata is not None, bname.decode()

                # Skip days, nominal burnup
                stream.read(struct.calcsize("2d"))

                (nnucs,) = struct.unpack("l", stream.read(struct.calcsize("l")))
                assert nnucs > 0, nnucs

                adens, mdens, _mbu = struct.unpack(
                    "3d", stream.read(struct.calcsize("3d")),
                )
                assert adens > 0, adens
                assert mdens > 0, mdens

                mdata["adens"] = adens
                mdata["mdens"] = mdens

                # Process isotopes
                buf = stream.read(struct.calcsize("ld") * nnucs)
                for z, _a in struct.iter_unpack("ld", buf):
                    if z > 0:
                        zais.add(z)

                buf = stream.read(struct.calcsize("l"))

        if self._allowedZAI is None:
            self._allowedZAI = zais
        else:
            self._allowedZAI.update(zais)

    def updateComps(self, compositions, timestep, threshold=0):
        assert self._names is not None
        assert self.compFile is not None

        if not timestep.coarse:
            raise ValueError(
                "This method should not be called for the first step. "
                "BOL compositions are already written in the input file. "
                "Call updateFromRestart if updates to ZAIs, atomic densities, "
                "or mass densites are needed."
            )

        if self._allowedZAI is None:
            self._allowedZAI = self._readZAI()

        day = timestep.currentTime / SECONDS_PER_DAY

        zais = [isotope.zai for isotope in compositions.isotopes]

        longDub = struct.calcsize("ld")

        with self.compFile.open("wb") as stream:
            for (bname, matdata), densities in zip(
                self._names.items(), compositions.densities,
            ):
                namelen = len(bname)
                stream.write(struct.pack("l", namelen))
                stream.write(struct.pack(f"{namelen}s", bname))
                stream.write(struct.pack("2d", self._FAKE_BURNUP, day))

                lost = 0.0
                total = 0.0
                isoZaiAdens = []
                for zai, adens in zip(zais, densities):
                    if adens < threshold or zai not in self._allowedZAI:
                        lost += adens
                        continue
                    isoZaiAdens.append((zai, adens))
                    total += adens

                # This assumes that the mass density is constant over
                # time, which is not true. Serpent uses the atom density
                # over the mass density in the transport routine, but keep
                # an eye on this

                stream.write(
                    struct.pack(
                        "l3d",
                        len(isoZaiAdens) + 1,
                        total,
                        matdata["mdens"],
                        self._FAKE_BURNUP,
                    )
                )

                buf = bytearray((1 + len(isoZaiAdens)) * longDub)
                struct.pack_into("ld", buf, 0, -1, lost)

                for count, (z, a) in enumerate(isoZaiAdens, start=1):
                    struct.pack_into("ld", buf, count * longDub, z, a)

                stream.write(buf)
