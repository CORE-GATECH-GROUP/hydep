.. _config:

Configuration
=============

:mod:`hydep` allows configuration through text files and through
the API. The :class:`hydep.Settings` class can be used to read in a 
text file, modify settings inside Python, and then apply the settings
to the overall simulation. This file is present to provide a look at
how the configuration file is structured, what the values represent,
and the allowable options.

.. _config-example:

Example configuration
---------------------

.. literalinclude:: ../hydep.cfg.example
    :language: cfg