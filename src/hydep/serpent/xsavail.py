r"""
Collection of homogenized data, including macroscopic cross
sections and diffusion coefficients available in Serpent
2.1.30 or greater. Aggregated using the following bash command::

    grep -o -e "^B1_.*(" -e "^INF_.*(" \
        $1 | sed -e s"/\s*(//" -e "s/^B1_//" -e "s/^INF_//" \
        | sort -u

Some flags have been removed, like the ``B1_CALCULATION``.

.. note::

    CMM diffusion and transport cross sections have been
    purposely removed, as they produce strange results
    when computed across sub-domains.

The following physics / values are not supported at this moment

*. Assembly discontinuity factors (``set adf``)
*. Assembly albedos (``set alb``)

"""
_SERPENT_GCUXS_NAMES = {
    "ABS",
    "CAPT",
    "CHID",
    "CHIP",
    "CHIT",
    "DIFFCOEF",
    "FISS",
    "FISS_FLX",
    "FLX",
    "I135_MICRO_ABS",
    "I135_YIELD",
    "INVV",
    "KAPPA",
    "KEFF",
    "KINF",
    "MICRO_FLX",
    "NSF",
    "NUBAR",
    "PM147_MICRO_ABS",
    "PM147_YIELD",
    "PM148_MICRO_ABS",
    "PM148M_MICRO_ABS",
    "PM148M_YIELD",
    "PM148_YIELD",
    "PM149_MICRO_ABS",
    "PM149_YIELD",
    "RABSXS",
    "REMXS",
    "S0",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "SCATT0",
    "SCATT1",
    "SCATT2",
    "SCATT3",
    "SCATT4",
    "SCATT5",
    "SCATT6",
    "SCATT7",
    "SCATTP0",
    "SCATTP1",
    "SCATTP2",
    "SCATTP3",
    "SCATTP4",
    "SCATTP5",
    "SCATTP6",
    "SCATTP7",
    "SM149_MACRO_ABS",
    "SM149_MICRO_ABS",
    "SM149_YIELD",
    "SP0",
    "SP1",
    "SP2",
    "SP3",
    "SP4",
    "SP5",
    "SP6",
    "SP7",
    "TOT",
    "TRANSPXS",
    "XE135_MACRO_ABS",
    "XE135_MICRO_ABS",
    "XE135_YIELD",
}

XS_2_1_30 = set()

while _SERPENT_GCUXS_NAMES:
    val = _SERPENT_GCUXS_NAMES.pop()
    if "_" not in val:
        XS_2_1_30.add(val.lower())
        continue
    tokens = val.split("_")
    XS_2_1_30.add("".join(
        [tokens[0].lower()] + [s.capitalize() for s in tokens[2:]]
    ))

del _SERPENT_GCUXS_NAMES

XS_2_1_30 = frozenset(XS_2_1_30)
