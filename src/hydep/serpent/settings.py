import os
import typing
import pathlib
import numbers

from hydep.settings import SubSetting

OptFile = typing.Optional[typing.Union[str, pathlib.Path]]
OptInt = typing.Optional[int]
OptFloat = typing.Optional[float]
PossiblePath = typing.Optional[pathlib.Path]


# TODO VER 3.7 dataclass
class SerpentSettings(SubSetting, sectionName="serpent"):
    """Main settings for Serpent solver

    All parameters are passed to corresponding attributes

    Attributes
    ----------
    datadir : pathlib.Path or None
        Directory of data files, like cross section libraries.
        If not given, will attempt to pull from ``SERPENT_DATA``
        environment variable. Libraries :attr:`acelib`,
        :attr:`declib`, and :attr:`nfylib` use this value if the
        initial paths do not point to files
    acelib : pathlib.Path or None
        Absolute path to cross section data file
    declib : pathlib.Path or None
        Absolute path to the Serpent decay file
    nfylib : pathlib.Path or None
        Absolute path to the Serpent neutron fission yield file
    sab : pathlib.Path or None
        Absolute path to the thermal scattering library file
    particles : int or None
        Number of particle to simulate per cycle
    generationsPerBatch : int or None
        Number of generations per batch
    active : int or None
        Number of active batches to simulate. Total number of active
        cycles will be ``active * generationsPerBatch``
    inactive : int or None
        Number of inative batches to simulate.
    seed : int or None
        Initial random number seed
    k0 : float or None
        Initial guess at multiplication factor. Defaults to 1.0, and must
        be bewteen zero and two
    executable : str or None
        Command to use when executing Serpent. Must be a valid shell
        command, e.g. ``"sss2"`` or ``"./sss2-custom"``
    omp : int or None
        Number of OMP threads to use. If initially given as ``None``, attempt
        to pull from ``OMP_NUM_THREADS`` environment variable
    mpi : int or None
        Number of MPI tasks to use

    """

    def __init__(
        self,
        # Writer settings
        datadir: OptFile = None,
        acelib: OptFile = None,
        declib: OptFile = None,
        nfylib: OptFile = None,
        sab: OptFile = None,
        particles: OptInt = None,
        generationsPerBatch: OptInt = None,
        active: OptInt = None,
        inactive: OptInt = None,
        seed: OptInt = None,
        k0: float = 1.0,
        # Runner settings
        executable: typing.Optional[str] = None,
        omp: OptInt = None,
        mpi: OptInt = None,
    ):
        if datadir is None:
            datadir = os.environ.get("SERPENT_DATA") or None
        self.datadir = datadir
        self.acelib = acelib
        self.declib = declib
        self.nfylib = nfylib
        self.sab = sab
        self.particles = particles
        self.generationsPerBatch = generationsPerBatch
        self.active = active
        self.inactive = inactive
        self.seed = seed
        self.k0 = k0
        self.executable = executable
        if omp is None:
            omp = os.environ.get("OMP_NUM_THREADS") or None
        self.omp = omp
        self.mpi = mpi

    @classmethod
    def pairs(cls) -> typing.Generator[typing.Tuple[str, str], None, None]:
        """Iterate over expected setting names and their attribute targets"""
        for attr in (
            "datadir",
            "acelib",
            "declib",
            "nfylib",
            "particles",
            "active",
            "inactive",
            "omp",
            "mpi",
            "executable",
            "k0",
            "seed",
        ):
            yield attr, attr
        for setting, attr in [
            ("generations per batch", "generationsPerBatch"),
            ("thermal scattering", "sab"),
        ]:
            yield setting, attr

    @property
    def datadir(self) -> PossiblePath:
        return self._datadir

    @datadir.setter
    def datadir(self, d: OptFile):
        if d is None:
            self._datadir = None
            return
        d = pathlib.Path(d).resolve()
        if not d.is_dir():
            raise NotADirectoryError(d)
        self._datadir = d

    def _validateLib(self, lib) -> pathlib.Path:
        lib = pathlib.Path(lib)
        if lib.is_file():
            return lib
        if self.datadir is not None:
            lib = self.datadir / lib
            if lib.is_file():
                return lib
        raise FileNotFoundError(lib)

    @property
    def acelib(self) -> PossiblePath:
        return self._acelib

    @acelib.setter
    def acelib(self, ace: OptFile):
        if ace is None:
            self._acelib = None
            return
        self._acelib = self._validateLib(ace)

    @property
    def declib(self) -> pathlib.Path:
        return self._declib

    @declib.setter
    def declib(self, dec: OptFile):
        if dec is None:
            self._declib = None
            return
        self._declib = self._validateLib(dec)

    @property
    def nfylib(self) -> PossiblePath:
        return self._nfylib

    @nfylib.setter
    def nfylib(self, nfy: OptFile):
        if nfy is None:
            self._nfylib = None
            return
        self._nfylib = self._validateLib(nfy)

    @property
    def sab(self) -> PossiblePath:
        return self._sab

    @sab.setter
    def sab(self, s: OptFile):
        if s is None:
            self._sab = None
            return
        s = pathlib.Path(s)
        if not s.is_file():
            raise FileNotFoundError(s)
        self._sab = s

    @property
    def particles(self) -> OptInt:
        return self._particles

    @particles.setter
    def particles(self, value: OptInt):
        if value is None:
            self._particles = None
            return
        self._particles = self.asPositiveInt("particles", value)

    @property
    def active(self) -> OptInt:
        return self._active

    @active.setter
    def active(self, value: OptInt):
        if value is None:
            self._active = None
            return
        self._active = self.asPositiveInt("active", value)

    @property
    def inactive(self) -> OptInt:
        return self._inactive

    @inactive.setter
    def inactive(self, value: OptInt):
        if value is None:
            self._inactive = None
            return
        self._inactive = self.asPositiveInt("inactive", value)

    @property
    def generationsPerBatch(self) -> OptInt:
        return self._generations

    @generationsPerBatch.setter
    def generationsPerBatch(self, value: OptInt):
        if value is None:
            self._generations = None
            return
        self._generations = self.asPositiveInt("generationsPerBatch", value)

    @property
    def seed(self) -> OptInt:
        return self._seed

    @seed.setter
    def seed(self, value: OptInt):
        if value is None:
            self._seed = None
            return
        self._seed = self.asPositiveInt("seed", value)

    @property
    def k0(self) -> OptFloat:
        return self._k0

    @k0.setter
    def k0(self, value: OptFloat):
        if value is None:
            self._k0 = None
            return
        if not isinstance(value, numbers.Real):
            value = float(value)
        if not (0 < value < 2):
            raise ValueError(value)
        self._k0 = value

    @property
    def executable(self) -> typing.Optional[str]:
        return self._executable

    @executable.setter
    def executable(self, exe: typing.Optional[str]):
        if exe is None:
            self._executable = None
            return
        if not isinstance(exe, str):
            raise TypeError(type(exe))
        self._executable = exe

    @property
    def omp(self) -> OptInt:
        return self._omp

    @omp.setter
    def omp(self, value: OptInt):
        if value is None:
            self._omp = None
            return
        self._omp = self.asPositiveInt("omp", value)

    @property
    def mpi(self) -> OptInt:
        return self._mpi

    @mpi.setter
    def mpi(self, value: OptInt):
        if value is None:
            self._mpi = None
            return
        self._mpi = self.asPositiveInt("mpi", value)

    def update(self, options: typing.Mapping[str, typing.Any]):
        """Update from a map of user supplied values

        All values map directly to attributes, with the following
        exceptions:

        *. ``"generations per batch"`` -> :attr:`generationsPerBatch`
        *. ``"thermal scattering"`` -> :attr:`sab`

        Parameters
        ----------
        options : mapping
            Keys are expected to be valid settings. Values will be
            set to attributes. Will be consumed with ``options.pop``

        Raises
        ------
        ValueError
            If any settings in ``options`` do not have a corresponding
            setting
        RuntimeError
            If a setting fails to be coerced to the expected data type

        """
        for setting, attr in self.pairs():
            value = options.pop(setting, None)
            if value is not None:
                try:
                    setattr(self, attr, value)
                except Exception as ee:
                    raise ee from RuntimeError(
                        f"Failed to coerce {setting} to allowable value from {value}"
                    )
        if options:
            remain = ", ".join(sorted(options))
            raise ValueError(
                f"The following Serpent settings were given, but "
                f"do not have corresponding attributes: {remain}",
            )
