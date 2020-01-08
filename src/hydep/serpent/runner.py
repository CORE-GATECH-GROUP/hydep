"""Serpent runner"""

import subprocess
import numbers

from hydep import FailedSolverError
from hydep.typed import BoundedTyped


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

    numOmp = BoundedTyped("numOmp", numbers.Integral, gt=0, allowNone=True)
    numMPI = BoundedTyped("numMPI", numbers.Integral, gt=0)

    def __init__(self, executable=None, numOMP=None, numMPI=None):
        self.executable = executable
        self.numOMP = numOMP
        self.numMPI = 1 if numMPI is None else numMPI

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

        if self.numOMP:
            cmd.extend("-omp {}".format(self.numOMP).split())

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
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode:
            msg = proc.stdout[-500:].decode() # TODO Improve error parsing
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
