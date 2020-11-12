hydep
=====

A hybrid depletion framework for reactor physics applications and
analysis. 

Installation
------------

The recommended approach is to install using a virtual environment.
From the command line, execute the command

.. code-block:: shell

    $ python -m venv /path/to/venv
    $ source /path/to/venv/bin/activate

The dependencies and package can then be installed with

.. code-block:: shell

    (venv) $ pip install .

The framework was designed to be a general and extensible tool, with
abstract interfaces for general transport codes. With this base
installation, one obtains the necessary classes to design concrete
interfaces, but no real useful interfaces. These are instead delegated
to extra packages

Serpent interface
~~~~~~~~~~~~~~~~~

The primary neutronics interface here is for the Serpent Monte Carlo
code [Serpent]_. The necessary dependencies can be installed with

.. code-block:: shell

    (venv) $ pip install .[serpent]

The ``hydep`` interfaces to Serpent will still be installed without this
command, but attempting to import from ``hydep.serpent`` will likely fail.

Two interfaces as presented, one for the current Serpent version and another
for an extended version of Serpent yet to be distributed. This later version
allows the framework to pass updated compositions to Serpent via a data file
without Serpent reloading the model and cross sections at each depletion step.
Discussion are being had regarding the best way to distribute these changes.
In the mean time, the `SerpentSolver`` can be used to perform hybrid
transport-depletion simulations, but with some waiting and restarting.

SFV interface
~~~~~~~~~~~~~

The primary reduced-order solver developed for this work is to the
spatial flux variation (SFV) method [SFV]_. This requires a small
FORTRAN library wrapped in a Python package. The source code for this
package is available at `CORE-GATECH-GROUP/sfv
<https://github.com/CORE-GATECH-GROUP/sfv>`_. Users can follow
`its installation instructions <https://github.com/CORE-GATECH-GROUP/sfv#installation>`_
or run the command

.. code-block:: shell

    (venv) $ pip install .[sfv]

.. note::

    Please be familiar with some of the issues uninstalling and upgrading
    this package, `presented here
    <https://github.com/CORE-GATECH-GROUP/sfv#upgrading--uninstalling>`_


Kitchen sink
~~~~~~~~~~~~

To install all dependencies for the base package, the Serpent interface, and
the SFV interface, the command

.. code-block:: shell

    (venv) $ pip install .[serpent,sfv]

Testing
-------

Tests require ``pytest`` which can be pulled from the ``test`` extras package

.. code-block:: shell

    (venv) $ pip install .[test]

Using ``pytest`` marks, parts of the library that relate to specific interfaces
can be excluded or isolated using the ``-m`` switch. The following will run
just tests related to the SFV interface

.. code-block:: shell

    (venv) $ pytest -m sfv

The current test layout does not well support testing just the base library with

.. code-block:: shell

    (venv) $ pytest -m "not serpent" -m "not sfv"

unless both the Serpent and SFV extras have been installed. Also, the Serpent
tests include a test with the coupled solver, which is not yet excluded with
a dedicated mark.


Documentation
-------------

Documentation is built using ``sphinx`` and currently is not hosted
online. It can be built locally by following these steps.

.. code-block:: shell

    (venv) $ pip install .[docs]
    (venv) $ cd docs
    (venv) $ make html

This will produce documentation that can be viewed locally by opening
``docs/_build/html/index.html``. The command ``make latexpdf`` will produce
``docs/_build/latex/hydep.pdf``.
There are two examples in ``docs/examples`` presented as jupyter notebooks that
build and simulate a 3-D pincell and then process the results.

Caveats / Warnings
------------------

This is a **highly experimental and developmental** library / tool.
While a modest set of cases are covered by tests and examples, there
are likely cases that are missed and could cause bugs. Please report
these and be forgiving.

This framework does not seek to be a general geometric modeling tool
for nuclear analysis, nor even a really good one. Some limits are
self imposed or assumed to move from developing geometry to
implementing the physics. First, the framework is primarily focused on
modeling Cartesian assemblies with annular fuel. This is sufficient for
most light water reactor analysis, as fuel pins are basically
concentric cylinders of materials, and these assemblies are regular
Cartesian lattices.

Second, many of the input files generated for transport solutions
should not be considered human readable. Some aesthetic considerations
have been taken, but universe names and identifiers may be hard to
understand. 

See ./docs/scope.rst for more description on the scope and limits of
this project.

References
----------

.. [Serpent] Leppanen, J. et al. "The Serpent Monte Carlo code: Status,
    development and applications in 2013." Ann. Nucl. Energy, `82 (2015) 142-150
    <http://www.sciencedirect.com/science/article/pii/S0306454914004095>`_

.. [SFV] Johnson, A. and Kotlyar, D. "A Transport-Free Method for Predicting
   the Post-depletion Spatial Neutron Flux." Nuc. Sci. Eng, `194 (2020) 120-137
   <https://doi.org/10.1080/00295639.2019.1661171>`_
