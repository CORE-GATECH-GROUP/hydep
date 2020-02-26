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

.. autodata: BARN_PER_CM2
    :annotation:

.. autodata: JOULES_PER_EV
    :annotation:

.. autodata: EV_PER_JOULE
    :annotation:

"""

import enum

SECONDS_PER_DAY = 86400  # Doc: Number of seconds per day
CM2_PER_BARN = 1e-24  # Doc: Number of squared centimeters per barn
BARN_PER_CM2 = 1 / CM2_PER_BARN  # Doc: Number of barns per squared cm
JOULES_PER_EV = 1.602176634e-19  # Doc: Joules per EV
EV_PER_JOULE = 1 / JOULES_PER_EV  # Doc: electron volts per Joule


class REACTION_MTS(enum.IntEnum):
    """Enumeration of various ENDF reaction MTs"""

    N_2N = 16
    N_3N = 17
    N_4N = 37
    TOTAL_FISSION = 18
    FIRST_CHANCE_FISSION = 19
    SECOND_CHANCE_FISSION = 20
    THIRD_CHANCE_FISSION = 21
    N_GAMMA = 102
    N_PROTON = 103
    N_ALPHA = 107


REACTION_MT_MAP = {
    "(n,2n)": REACTION_MTS.N_2N,
    "(n,3n)": REACTION_MTS.N_3N,
    "(n,4n)": REACTION_MTS.N_4N,
    "(n,gamma)": REACTION_MTS.N_GAMMA,
    "(n,p)": REACTION_MTS.N_PROTON,
    "(n,a)": REACTION_MTS.N_ALPHA,
    "fission": REACTION_MTS.TOTAL_FISSION,
}


FISSION_REACTIONS = frozenset(
    {
        REACTION_MTS.TOTAL_FISSION,
        REACTION_MTS.FIRST_CHANCE_FISSION,
        REACTION_MTS.SECOND_CHANCE_FISSION,
        REACTION_MTS.THIRD_CHANCE_FISSION,
    }
)
