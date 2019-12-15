"""
Shadow of OpenMC depletion base classes

Copyright: 2011-2019 Massachusetts Institute of Technology and
OpenMC collaborators

https://docs.openmc.org
https://docs.openmc.org/en/stable/pythonapi/deplete.html
https://github.com/openmc-dev/openmc

This module should only be imported if ``openmc.deplete`` cannot
be imported.
"""

from abc import ABC, abstractmethod


class DepSystemSolver(ABC):
    r"""Abstract class for solving depletion equations

    Responsible for solving

    .. math::

        \frac{\partial \vec{N}}{\partial t} = \bar{A}\vec{N}(t),

    for :math:`0< t\leq t +\Delta t`, given :math:`\vec{N}(0) = \vec{N}_0`

    """

    @abstractmethod
    def __call__(self, A, n0, dt):
        """Solve the linear system of equations for depletion

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            Sparse transmutation matrix ``A[j, i]`` desribing rates at
            which isotope ``i`` transmutes to isotope ``j``
        n0 : numpy.ndarray
            Initial compositions, typically given in number of atoms in some
            material or an atom density
        dt : float
            Time [s] of the specific interval to be solved

        Returns
        -------
        numpy.ndarray
            Final compositions after ``dt``. Should be of identical shape
            to ``n0``.

        """
