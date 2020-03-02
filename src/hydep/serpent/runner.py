"""Serpent runner"""

import logging
import subprocess
import numbers
import os

from hydep import FailedSolverError


class BaseRunner:
    """Class responsible for running Serpent

    Parameters
    ----------
    executable : str, optional
        Serpent executable. Can either be the command name
        or a path to the executable
    omp : int, optional
        Number of OMP threads to use. Will pass to the
        ``-omp`` command line argument. Use a value of ``None``
        if this should be determined by Serpent using the
        ``OMP_NUM_THREADS`` environment variable
    mpi : int, optional
        Number of MPI tasks to use when running Serpent. If provided,
        Serpent will be run using ``mpirun -np <N>``

    Attributes
    ----------
    executable : str or None
        Serpent executable. Can either be the command name
        or a path to the executable. Must be provided prior
        to :meth:`__call__`
    omp : int or None
        Number of OMP threads to use. Will pass to the
        ``-omp`` command line argument. Use a value of ``None``
        if this should be determined by Serpent using the
        ``OMP_NUM_THREADS`` environment variable
    mpi : int or None
        Number of MPI tasks to use when running Serpent. If provided,
        Serpent will be run using ``mpirun -np <N>``

    """
    def __init__(self, executable=None, omp=None, mpi=None):
        self.executable = executable
        self.omp = omp
        self.mpi = mpi

    @property
    def omp(self):
        return self._omp

    @omp.setter
    def omp(self, value):
        if value is None:
            value = int(os.environ.get("OMP_NUM_THREADS", 1))
        if not isinstance(value, numbers.Integral):
            raise TypeError(f"Cannot set number of OMP threads to {value}: not integer")
        elif value < 1:
            raise ValueError(
                f"Cannot set number of OMP threads to {value}: not positive"
            )
        self._omp = value

    @property
    def mpi(self):
        return self._mpi

    @mpi.setter
    def mpi(self, value):
        if value is None:
            self._mpi = 1
        elif not isinstance(value, numbers.Integral):
            raise TypeError(f"Cannot set number of MPI tasks to {value}: not integer")
        elif value < 1:
            raise ValueError(f"Cannot set number of MPI tasks to {value}: not positive")
        else:
            self._mpi = value

    @property
    def executable(self):
        return self._executable

    @executable.setter
    def executable(self, value):
        # This could be a path to an executable or a name of the executable
        # as it can be found in PATH. Since it must be set before makeCmd
        # and a non-existent command will cause __call__ to fail, don't
        # perform checks
        self._executable = value

    def makeCommand(self):
        """Create a list of arguments given settings

        Returns
        -------
        list
            List of strings that can be passed to
            :func:`subprocess.run`. Relies on :attr:`numOMP` and
            :attr:`numMPI` to populate the list

        Examples
        --------
        >>> r = BaseRunner("sss2")
        >>> r.makeCmd()
        ['sss2']
        >>> r.mpi = 2
        >>> r.makeCommand()
        ['mpirun', '-np', '2', 'sss2']
        >>> r.omp = 12
        >>> r.makeCommand()
        ['mpirun', '-np', '2', 'sss2', '-omp', '12']

        Raises
        ------
        AttributeError
            If :attr:`executable` is not configured

        """
        if self.executable is None:
            raise AttributeError(
                "Serpent executable not configured for {}".format(self)
            )
        if self.mpi > 1:
            # TODO Machinefile?
            cmd = "mpirun -np {}".format(self.mpi).split()
        else:
            cmd = []

        cmd.append(self.executable)

        if self.omp > 1:
            cmd.extend("-omp {}".format(self.omp).split())

        logging.getLogger("hydep.serpent").debug(f"Executable commands: {cmd}")

        return cmd

    def configure(self, section):
        """Configure this runner using the ``hydep.serpent`` section

        Parameters
        ----------
        section : configparser.SectionProxy
            Serpent specific options

        """
        omp = section.getint("omp")
        if omp is not None:
            self.omp = omp

        mpi = section.getint("mpi")
        if mpi is not None:
            self.mpi = mpi

        executable = section.get("executable")
        if executable is not None:
            self.executable = executable


class SerpentRunner(BaseRunner):
    """Class responsible for running a Serpent job

    Parameters
    ----------
    executable : str, optional
        Serpent executable. Can either be the command name
        or a path to the executable
    omp : int, optional
        Number of OMP threads to use. Will pass to the
        ``-omp`` command line argument. Use a value of ``None``
        if this should be determined by Serpent using the
        ``OMP_NUM_THREADS`` environment variable
    mpi : int, optional
        Number of MPI tasks to use when running Serpent. If provided,
        Serpent will be run using ``mpirun -np <N>``

    Attributes
    ----------
    executable : str or None
        Serpent executable. Can either be the command name
        or a path to the executable. Must be provided prior
        to :meth:`__call__`
    omp : int or None
        Number of OMP threads to use. Will pass to the
        ``-omp`` command line argument. Use a value of ``None``
        if this should be determined by Serpent using the
        ``OMP_NUM_THREADS`` environment variable
    mpi : int or None
        Number of MPI tasks to use when running Serpent. If provided,
        Serpent will be run using ``mpirun -np <N>``

    """

    def __call__(self, inputpath):
        """Run Serpent given this input file

        Parameters
        ----------
        inputfile : pathlib.Path
            Path to existing input file.

        Raises
        ------
        hydep.FailedSolverError
            Error message contains the tail end of the Serpent
            output. Not well parsed

        """
        if not inputpath.is_file():
            raise IOError("Input file {} does not exist".format(inputpath))

        cmd = self.makeCommand() + [inputpath]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode:
            msg = proc.stdout[-500:].decode()
            # TODO Improve error parsing?
            raise FailedSolverError(msg)



