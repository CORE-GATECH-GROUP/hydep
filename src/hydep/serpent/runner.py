"""Serpent runner"""

import logging
import subprocess
import numbers

from hydep import FailedSolverError


class SerpentRunner:
    """Class responsible for running a Serpent job

    Parameters
    ----------
    executable : str, optional
        Serpent executable. Can either be the command name
        or a path to the executable
    numOmp : int, optional
        Number of OMP threads to use. Will pass to the
        ``-omp`` command line argument. Use a value of ``None``
        if this should be determined by Serpent using the
        ``OMP_NUM_THREADS`` environment variable
    numMPI : int, optional
        Number of MPI tasks to use when running Serpent. If provided,
        Serpent will be run using ``mpirun -np <N>``

    Attributes
    ----------
    executable : str or None
        Serpent executable. Can either be the command name
        or a path to the executable. Must be provided prior
        to :meth:`__call__`
    numOmp : int or None
        Number of OMP threads to use. Will pass to the
        ``-omp`` command line argument. Use a value of ``None``
        if this should be determined by Serpent using the
        ``OMP_NUM_THREADS`` environment variable
    numMPI : int or None
        Number of MPI tasks to use when running Serpent. If provided,
        Serpent will be run using ``mpirun -np <N>``

    """

    def __init__(self, executable=None, numOMP=None, numMPI=None):
        self.executable = executable
        self.numOMP = numOMP
        self.numMPI = numMPI
    @property
    def numOMP(self):
        return self._numOMP

    @numOMP.setter
    def numOMP(self, value):
        if value is None:
            value = 1
        if not isinstance(value, numbers.Integral):
            raise TypeError(f"Cannot set number of OMP threads to {value}: not integer")
        elif value < 1:
            raise ValueError(
                f"Cannot set number of OMP threads to {value}: not positive"
            )
        self._numOMP = value

    @property
    def numMPI(self):
        return self._numMPI

    @numMPI.setter
    def numMPI(self, value):
        if value is None:
            self._numMPI = 1
        elif not isinstance(value, numbers.Integral):
            raise TypeError(f"Cannot set number of MPI tasks to {value}: not integer")
        elif value < 1:
            raise ValueError(f"Cannot set number of MPI tasks to {value}: not positive")
        else:
            self._numMPI = value

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

    def makeCmd(self):
        """Create a list of arguments given settings

        Returns
        -------
        list
            List of strings that can be passed to
            :func:`subprocess.run`. Relies on :attr:`numOMP` and
            :attr:`numMPI` to populate the list

        Examples
        --------
        >>> r = SerpentRunner("sss2")
        >>> r.makeCmd()
        ['sss2']
        >>> r.numMPI = 2
        >>> r.makeCmd()
        ['mpirun', '-np', '2', 'sss2']
        >>> r.numOMP = 12
        >>> r.makeCmd()
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
        if self.numMPI > 1:
            # TODO Machinefile?
            cmd = "mpirun -np {}".format(self.numMPI).split()
        else:
            cmd = []

        cmd.append(self.executable)

        if self.numOMP > 1:
            cmd.extend("-omp {}".format(self.numOMP).split())

        logging.getLogger("hydep.serpent").debug(f"Executable commands: {cmd}")

        return cmd

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

        cmd = self.makeCmd() + [inputpath]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode:
            msg = proc.stdout[-500:].decode()
            # TODO Improve error parsing?
            raise FailedSolverError(msg)

    def configure(self, section):
        """Configure this runner using the ``hydep.serpent`` section

        Parameters
        ----------
        section : configparser.SectionProxy
            Serpent specific options

        """
        omp = section.getint("omp")
        if omp is not None:
            self.numOMP = omp

        mpi = section.getint("mpi")
        if mpi is not None:
            self.numMPI = mpi

        executable = section.get("executable")
        if executable is not None:
            self.executable = executable
