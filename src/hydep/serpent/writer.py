"""
Serpent writer
"""

import pathlib
import warnings
from collections import deque

import numpy

import hydep
from hydep.internal import getIsotope
from hydep.typed import TypedAttr, IterableOf
import hydep.internal.features as hdfeat


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
    """

    _temps = (300, 600, 900, 1200, 1500)
    # TODO Allow config control over default material temperature
    _defaulttemp = 600
    # TODO Allow config control over ace, dec, nfy libraries
    acelib = "sss_endfb7u.xsdata"
    declib = "sss_endfb7.dec"
    nfylib = "sss_endfb7.nfy"
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
        if not self.options:
            # TODO Improve - only really need pop?
            raise AttributeError("Not well configured")

        self.base = self._setupfile(path)

        with self.base.open("w") as stream:
            self._writematerials(stream)
            self._writegeometry(stream)
            self._writesettings(stream)
            if self.hooks:
                self._writehooks(stream)

    def _writesettings(self, stream):
        self.commentblock(stream, "BEGIN SETTINGS BLOCK")
        for attr in ["acelib", "declib", "nfylib"]:
            stream.write('set {} "{}"\n'.format(attr, getattr(self, attr)))
        stream.write("set pop {} ".format(self.options["particles"]))

        gen = self.options["generations per batch"]
        active = self.options["active"]
        skipped = self.options["skipped"]
        stream.write("{} {}".format(gen * active, gen * skipped))
        stream.write(" {:.5f} {}\n".format(self.options.get("k0", 1.0), gen))

        stream.write("set bc")
        for value in self.options.get("bc", [1]):
            stream.write(" " + str(value))
        stream.write("\n")

        seed = self.options.get("seed")
        if seed is not None:
            stream.write("set seed {}\n".format(seed))

        stream.write("""% Hard set one group [0, 20] MeV for all data
ene {grid} 2 1 0 20
set nfg {grid}
""".format(grid=self._eneGridName))

    def _writematerials(self, stream):
        self.commentblock(stream, "BEGIN MATERIAL BLOCK")
        for mat in self.model.root.findMaterials():
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
            stream.write("% {}\n".format(material.name))
        stream.write(
            "mat {} {}{:<9.7f}".format(
                material.id,
                "-" if material.mdens is not None else "",
                material.adens or material.mdens,
            )
        )
        if material.volume is not None:
            stream.write(" vol {:9.7f}".format(material.volume))
        if (material.temperature is not None
                and material.temperature not in self._temps):
            stream.write(" tmp {:9.7f}".format(material.temperature))

        if isinstance(material, hydep.BurnableMaterial):
            stream.write(" burn 1")

        stream.write("\n")
        tlib = self._getmatlib(material)

        for isotope, adens in sorted(material.items()):
            # TODO Metastable
            stream.write(
                "{}{:03}.{} {:13.9E}\n".format(*isotope.triplet[:2], tlib, adens)
            )

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
        self._writecellbounds(stream, self.model.root, rootid, 0)

    def writeUniverse(self, stream, u, memo):
        """Write the geometry definition for this material

        Parameters
        ----------
        stream : writable
            Object onto which to write the geometry
        u : hydep.Universe
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

        writeas = "p" + str(pin.id)
        memo[pin.id] = writeas

        if pin.name is not None:
            stream.write("% {}\n".format(pin.name))

        if any(isinstance(m, hydep.BurnableMaterial) for m in pin.materials):
            self._writeburnablepin(stream, pin, writeas)
        else:
            stream.write("pin {}\n".format(writeas))
            for r, m in zip(pin.radii, pin.materials):
                stream.write("{} {:.7f}\n".format(m.id, r))
            stream.write(str(pin.outer.id) + "\n")
        stream.write("\n")
        return writeas

    @staticmethod
    def _getPinRadiusID(pin, ix):
        # TODO Convert to a class attribute and formatter?
        return "{}_r{}".format(pin.id, ix)

    def _writeburnablepin(self, stream, pin, writeas):
        # TODO Write a single surface for each unique radius?
        surfaces = deque(maxlen=2)  # [lower surf, outer surf]
        for ix, (r, m) in enumerate(pin):
            surfaces.append(self._getPinRadiusID(pin, ix))
            if isinstance(m, hydep.BurnableMaterial):
                # Write an infinite universe of this material
                stream.write("""surf {surf}_i inf
cell {surf}_i {uid} {uid} -{surf}
""".format(surf=surfaces[-1], uid=m.id))
                filler = "fill {}".format(m.id)
            else:
                filler = m.id

            if r < numpy.inf:
                stream.write("surf {} cyl 0.0 0.0 {:.5f}\n".format(surfaces[-1], r))
                stream.write(
                    "cell {surf} {pid} {fill} ".format(
                        surf=surfaces[-1], pid=writeas, fill=filler
                    )
                )
                if ix:
                    stream.write("{} -{}\n".format(*surfaces))
                else:
                    stream.write("-{}\n".format(surfaces[0]))
            else:
                stream.write(
                    "cell {outer} {pid} {fill} {inner}\n".format(
                        outer=surfaces[1], inner=surfaces[0], pid=writeas, fill=filler
                    )
                )

    def _writelattice(self, stream, lat, memo):
        previous = memo.get(lat.id)
        if previous is not None:
            return previous

        outermost = "cl" + str(lat.id)
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
            stream.write("% {}\n".format(lat.name))

        stream.write(
            "lat {name} 1 0.0 0.0 {nx} {ny} {pitch:.5f}\n".format(
                name=innermost, nx=lat.nx, ny=lat.ny, pitch=lat.pitch
            )
        )
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
            surf = "rect {xy}".format(lid=universe.id, xy=xybounds)
        else:
            surf = "cuboid {xy} {mnz:.5f} {mxz:.5f}".format(
                xy=xybounds, mnz=bounds.z[0], mxz=bounds.z[1]
            )
        stream.write(
            """
surf {lid}_x {surf}
cell {lid}_1 {u} fill {filler} -{lid}_x
cell {lid}_2 {u} {outer} {lid}_x
""".format(
                surf=surf,
                lid=universe.id,
                filler=filler,
                outer=outer,
                u=universenumber,
            )
        )

    def _writestack(self, stream, lstack, memo):
        previous = memo.get(lstack.id)
        if previous is not None:
            return previous

        writeas = "ls" + str(lstack.id)

        memo[lstack.id] = writeas

        subids = []
        for item in lstack:
            uid = memo.get(item.id)
            if uid is None:
                uid = self.writeUniverse(stream, item, memo)
                memo[item.id] = uid
            subids.append(uid)

        if lstack.name is not None:
            stream.write("% {}\n".format(lstack.name))

        stream.write("lat {} 9 0.0 0.0 {}\n".format(writeas, lstack.nLayers))
        for lower, sub in zip(lstack.heights[:-1], subids):
            stream.write("{:.5f} {}\n".format(lower, sub))

        return writeas

    def _writeInfMaterial(self, stream, infmat, memo):
        previous = memo.get(infmat.id)
        if previous is not None:
            return previous

        writeas = "inf" + str(infmat.id)
        memo[infmat.id] = writeas

        if infmat.material.name is not None:
            stream.write("% Infinite region filled with {}\n".format(
                infmat.material.name))

        stream.write("""surf {writeas} inf
cell {writeas} {writeas} {mid} -{writeas}
""".format(writeas=writeas, mid=infmat.material.id))

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

    def _parseboundaryconditions(self, conds):
        bc = []
        if len(conds) == 1:
            conds = conds * 3
        for c, dim in zip(conds, ["x", "y", "z"]):
            bcval = self.bcmap.get(c)
            if bcval is None:
                raise ValueError(
                    "Unsure how to process boundary condition {}. "
                    "Supported values are {}".format(c, ", ".join(self.bcmap))
                )
            bc.append(bcval)
        self.options["bc"] = bc

    def _writeIterableOverLines(self, stream, lines, delim=" "):
        for count, line in enumerate(lines):
            stream.write(line)
            if count and count % self._groupby == 0:
                stream.write("\n")
            else:
                stream.write(delim)
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
        stream.write("det flux de {}\n".format(self._eneGridName))
        lines = map("du {}", (m.id for m in self.burnable))
        self._writeIterableOverLines(stream, lines)

    def _writelocalmicroxs(self, stream):
        self.commentblock(stream, """BEGIN MICROSCOPIC REACTION XS BLOCK
Need to trick Serpent into given this information, but we don't want a ton
of depletion. Add a single one day step here. Maybe hack something later""")
        stream.write("dep daystep 1\n")
        for m in self.burnable:
            stream.write("set mdep {mid} 1.0 1 {mid}\n".format(mid=m.id))
            reactions = self._getReactions(set(m))
            lines = ("{} {}".format(z, m) for z, m in sorted(reactions))
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
            for reaction in iso.reactions:
                if reaction.target not in previous:
                    isotopes.add(reaction.target)
                reactions.add((iso.zai, reaction.mt))
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
            stream.write("set power {:.7E}\n".format(power))

            for m in self.burnable:
                self.writemat(stream, m)

        return steadystate
