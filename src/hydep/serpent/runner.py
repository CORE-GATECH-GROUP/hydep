"""Serpent runner"""

import pathlib
import time
import logging
import subprocess
import numbers
import os
import re
from enum import Enum, auto
import signal

from hydep import FailedSolverError

# Patterns used to search for the source of error in
# the output stream. Should be binary raw strings.
# Ordering might be important? Most likely errors first?

ERROR_PATTERNS = [
    rb"Out of memory error\s+\(.*\)",
]


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
        # as it can be found in PATH. Since it must be set before makeCommand
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
        >>> r.makeCommand()
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

    def configure(self, settings):
        """Configure the runner using user-provided settings

        If any of the following attributes on ``settings`` exist and
        are not ``None``, they will be mapped directly to attributes
        on this instance:

        * ``executable`` -> :attr:`executable`
        * ``omp`` -> :attr:`omp`
        * ``mpi`` -> :attr:`mpi`

        Parameters
        ----------
        settings : hydep.serpent.SerpentSettings
            Serpent specific settings

        """
        for attr in {"executable", "omp", "mpi"}:
            value = getattr(settings, attr, None)
            if value is not None:
                setattr(self, attr, value)

    def _reportFailure(self, tail: bytes):
        msg = None
        for pattern in ERROR_PATTERNS:
            match = re.search(pattern, tail)
            if match is not None:
                start, end = match.span(0)
                msg = f"Reason:\n{tail[start:end].decode()}"
                break
        else:
            msg = f"Last portion of stdout:\n{tail.decode()}"
        raise FailedSolverError(f"Serpent has failed. {msg}")


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
            self._reportFailure(proc.stdout[-500:])


class STATE(Enum):
    """Enumerations of various states for the external coupling"""
    INACTIVE = auto()  #: State prior to starting the coupling
    RUNNING = auto()  #: Serpent is currently running
    WAIT_IFC = auto()  #: Transport has completed, waiting for interfaces
    WAIT_NEXT_STEP = auto()  #: Waiting before next depletion step
    TERM = auto()  #: Serpent has signaled that it is terminating
    DEAD = auto()  #: Process has closed, possibly because it died


class ExtDepRunner(BaseRunner):
    """Class for running Serpent via the external depletion interface"""
    def __init__(self, executable=None, omp=None, mpi=None):
        super().__init__(executable, omp, mpi)
        self._proc = None
        self._state = STATE.INACTIVE
        self._output = None

    @property
    def state(self):
        return self._state

    def _signalHandler(self, insig, stack):
        if insig == signal.SIGUSR1:
            self._state = STATE.WAIT_IFC
        elif insig == signal.SIGUSR2:
            self._state = STATE.WAIT_NEXT_STEP
        elif insig == signal.SIGTERM:
            self._state = STATE.TERM

    def terminate(self):
        """Terminate the Serpent process and any output pipes"""
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        if self._output is not None:
            self._output.close()
            self._output = None

    def __del__(self):
        """Ensure the runner and connections are tidied up"""
        self.terminate()

    def start(self, inputfile, output=None):
        """Start coupled Serpent and run the first transport step

        Parameters
        ----------
        inputfile : str or pathlib.Path
            Path to Serpent input file
        output : str or pathlib.Path or writable, optional
            Destination for the output stream. If a string of path given,
            write all outputs to a file with that name. If ``None``,
            capture output internally. Otherwise, the object must
            be writable, as all output from Serpent will be forwarded
            to this location

        """
        cmd = self.makeCommand() + [inputfile]

        if output is None:
            output = subprocess.PIPE
        elif isinstance(output, (str, pathlib.Path)):
            output = self._output = open(output, "w")
        else:
            assert hasattr(output, "write")

        signal.signal(signal.SIGUSR1, self._signalHandler)
        signal.signal(signal.SIGUSR2, self._signalHandler)
        signal.signal(signal.SIGTERM, self._signalHandler)

        self._proc = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
        self._state = STATE.RUNNING
        self._wait(STATE.WAIT_IFC)

    def _wait(self, desiredState):
        stat = self._proc.poll()
        while stat is None:
            if self.state == desiredState:
                break
            time.sleep(1)
            stat = self._proc.poll()
        else:
            self._state = STATE.DEAD
            # Avoid a race-condition where Serpent has told us it is
            # terminating, and did so before we could check
            # the state from the signal handler. If we wanted Serpent
            # to terminate, then this should be okay
            if desiredState == STATE.TERM:
                return
            # All is lost
            self._reportFailure(self._proc.stdout[-500:])

    def solveNext(self):
        """Tell Serpent to solve the next transport step

        Assumes that the compositions have already been written
        according to the external depletion interface commands.

        Serpent is first told to advance to the next point in
        time. Serpent sends a signal that it is ready for the next
        stage, and that any interfaces should be updated. In the
        custom version of Serpent that presents this setting, the
        depletion interface file is only written at the first stage.

        Externally, through this framework, the compositions are
        updated prior to calling this method. The second signal tells
        Serpent the interfaces are ready and the next transport step
        should be taken.

        """
        self._proc.send_signal(signal.SIGUSR2)
        self._wait(STATE.WAIT_NEXT_STEP)

        self._proc.send_signal(signal.SIGUSR1)
        self._state = STATE.RUNNING
        self._wait(STATE.WAIT_IFC)

    def solveEOL(self):
        """Advance to and solve the final point in time

        Performs the same actions as :meth:`solveNext`, and then
        waits for Serpent to terminate.
        """
        self.solveNext()
        self._proc.send_signal(signal.SIGUSR2)
        self._wait(STATE.TERM)
