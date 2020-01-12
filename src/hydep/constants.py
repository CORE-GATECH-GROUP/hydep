"""
Constants module
================

.. warning::

    There is no way to make these truly constant, e.g.
    non-writeable. **Don't alter these values**, or else
    your simulation will be wrong.

Values that are expressed as ``X_PER_Y`` can be used to
convert units of ``Y`` to units of ``X`` by multiplication. To
get the number of seconds in five days, one would perform the
below conversion

>>> days = 5
>>> "{:.2E}".format(days * SECONDS_PER_DAY)
'4.32E+05'

Constants
---------

.. autodata: DAY_PER_SECOND
    :annotation:

.. autodata: CM2_PER_BARN
    :annotation:

"""
SECONDS_PER_DAY = 86400  # Doc: Number of seconds per day
CM2_PER_BARN = 1e-24  # Doc: Number of squared centimeters per barn
