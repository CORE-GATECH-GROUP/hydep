"""
Serpent writer
"""

import pathlib
import warnings

import numpy

import hydep
from hydep.typed import TypedAttr, IterableOf


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
    hooks : Set[hydep.features.Feature]
        Each entry indicates a specific type of physics that
        must be run. Including :data:`hydep.features.FISSION_MATRIX`
        indicates that a fission matrix containing the universes
        of all burnable materials must be created.
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

    def __init__(self):
        self.model = None
        self.burnable = None
        self.base = None
        self.steadystatefile = None
        self.hooks = set()
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
        for attr in {"acelib", "declib", "nfylib"}:
            stream.write('set {} "{}"\n'.format(attr, getattr(self, attr)))
        stream.write("set pop {} ".format(self.options["particles"]))

        gen = self.options["generations per batch"]
        active = self.options["active"]
        skipped = self.options["skipped"]
        stream.write("{} {}".format(gen * active, gen * skipped))
        stream.write(" {:7.5f} {}\n".format(self.options.get("k0", 1.0), gen))

        stream.write("set bc")
        for value in self.options.get("bc", [1]):
            stream.write(" " + str(value))
        stream.write("\n")

    def _writematerials(self, stream):
        self.commentblock(stream, "BEGIN MATERIAL BLOCK")
        for mat in self.model.findMaterials():
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
        # TODO Some guard against name clashes / invalid Serpent names
        # Material named "ss 304" would break / cause weird events
        # Replace " " with "_"?
        stream.write(
            "mat {} {}{:<10.7f}".format(
                material.name,
                "-" if material.mdens is not None else "",
                material.adens or material.mdens,
            )
        )
        if material.volume is not None:
            stream.write(" vol {:10.7f}".format(material.volume))
        if material.temperature is not None:
            stream.write(" tmp {:10.7f}".format(material.temperature))

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
                    "Temperature {:7.3f} for {} too low. Using {}".format(
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
        self.writeUniverse(stream, self.model.root, {})
        self._writecellbounds(stream, self.model.root, self.model.root.id, 0)

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
        raise TypeError(type(u))

    @staticmethod
    def _writepin(stream, pin, memo):
        previous = memo.get(pin.id)
        if previous is not None:
            return previous
        memo[pin.id] = name = pin.name or pin.id
        stream.write("pin {}\n".format(name))
        for r, m in zip(pin.radii, pin.materials):
            stream.write("{} {:10.7f}\n".format(m.name, r))
        stream.write(pin.outer.name + "\n")
        return name

    def _writelattice(self, stream, lat, memo):
        previous = memo.get(lat.id)
        if previous is not None:
            return previous

        memo[lat.id] = lat.id
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

        if lat.outer is None:
            latname = lat.id
        else:
            latname = lat.id + "_0"

        stream.write(
            "lat {name} 1 0.0 0.0 {nx} {ny} {pitch:7.5f}\n".format(
                name=latname, nx=lat.nx, ny=lat.ny, pitch=lat.pitch
            )
        )
        while univrows:
            stream.write(" ".join(map(str, univrows.pop())) + "\n")

        if lat.outer is not None:
            self._writecellbounds(stream, lat, latname, lat.id, lat.outer.name)

        return lat.id

    @staticmethod
    def _writecellbounds(stream, universe, filler, universenumber, outer="outside"):
        bounds = universe.bounds
        xybounds = " ".join(
            map("{:7.5f}".format, (bounds.x[0], bounds.x[1], bounds.y[0], bounds.y[1]),)
        )
        if bounds.z is None or (-bounds.z[0] == bounds.z[1] == numpy.inf):
            surf = "rect {xy}".format(lid=universe.id, xy=xybounds)
        else:
            surf = "cuboid {xy} {mnz:7.5f} {mxz:7.5f}".format(
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
        subids = []
        for item in lstack:
            uid = memo.get(item.id)
            if uid is None:
                uid = self.writeUniverse(stream, item, memo)
                memo[item.id] = uid
            subids.append(uid)
        stream.write("lat {} 9 0.0 0.0 {}\n".format(lstack.id, lstack.nLayers))
        for lower, sub in zip(lstack.heights[:-1], subids):
            stream.write("{:7.5f} {}\n".format(lower, sub))
        memo[lstack.id] = lstack.id
        return lstack.id

    def configure(self, options, section, level):
        """Configure the writer

        Parameters
        ----------
        options : configparser.ConfigParser
            Collection of all user-specified options. Some may not apply
            to this writer.
        section : str
            Specific sub-section to be processed. Guarunteed to apply
            to this writer.
        level : int
            Depth or specificity. Currently support three levels:
            ``hydep``, ``hydep.montecarlo``, and ``hydep.serpent``.
            Options like verbosity and initial :math:`k` will be
            read read from level ``0``, with more specifics in levels
            ``1`` and ``2``

        """
        k0 = options.getfloat(section, "initial k", fallback=None)
        if k0 is not None:
            self.options["k0"] = k0

        bc = options.get(section, "boundary conditions", fallback=None)
        if bc is not None:
            self._parseboundaryconditions(bc.replace(",", " ").split())

        if level == 0:
            return

        for key in [
            "particles",
            "generations per batch",
            "active",
            "skipped",
        ]:
            value = options.getint(section, key, fallback=None)
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

    def _writehooks(self, stream):
        self.commentblock(stream, "BEGIN HOOKS")
        if hydep.features.FISSION_MATRIX in self.hooks:
            self._writefmtx(stream)
        if hydep.features.HOMOG_LOCAL in self.hooks:
            self._writelocalgcu(stream)
        if hydep.features.MICRO_REACTION_XS in self.hooks:
            self._writelocalmicroxs(stream)

    def _writefmtx(self, stream):
        stream.write("set fmtx 1 ")
        for count, uid in enumerate(m.id for m in self.burnable):
            stream.write("{} ".format(uid))
            if count and count % self._groupby == 0:
                stream.write("\n")
        stream.write("\n")

    def _writelocalgcu(self, stream):
        stream.write("set gcu ")
        for count, uid in enumerate(m.id for m in self.burnable):
            stream.write("{} ".format(uid))
            if count and count % self._groupby == 0:
                stream.write("\n")
        stream.write("\n")

    def _writelocalmicroxs(self, stream):
        pass

    def writeSteadyStateFile(self, path, timestep):
        if self.burnable is None:
            raise AttributeError("No burnable material ordering set on {}".format(self))
        if self.base is None:
            raise AttributeError(
                "Base file to be included not found on {}".format(self)
            )

        self.steadystate = self._setupfile(path)
        with self.steadystate.open("w") as stream:
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

            for m in self.burnable:
                self.writemat(stream, m)
