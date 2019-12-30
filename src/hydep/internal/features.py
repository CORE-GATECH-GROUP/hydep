"""Basic feature management system

Used to indicate what features / physics are needed to
couple the reduced order code to the high fidelity code
"""

from collections import namedtuple


Feature = namedtuple("Feature", ["name", "description"])


FISSION_MATRIX = Feature("fission matrix", "the fission matrix")
HOMOG_GLOBAL = Feature(
    "global homogenization",
    "homogenized macroscopic group constants across the entire domain",
)
HOMOG_LOCAL = Feature(
    "local homogenization",
    "homogenized macroscopic group constants across arbitrary sub-domain",
)
MICRO_REACTION_XS = Feature(
    "microscopic cross sections",
    "microscopic reaction cross sections across arbitrary sub-domain",
)
