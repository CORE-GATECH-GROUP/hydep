from pkg_resources import parse_version
import pkg_resources  # distributed with setuptools

_expects = parse_version("0.3.1")

try:
    import sfv

    if parse_version(sfv.__version__) < _expects:
        raise ValueError(
            f"Expected version {_expects} for sfv, found {sfv.__version__}"
        )
except ImportError:
    raise ImportError("Reach out to developers for the sfv package")

del _expects, parse_version

from sfv import applySFV, getAdjFwdEig
