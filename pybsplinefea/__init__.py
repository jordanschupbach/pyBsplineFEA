"""
The pyBsplineFEA library.
"""

__name__ = "pybsplinefea"

from ._version import __version__
from .base_algos import *
from .benchmarks import *
from .bspline_mse import BSplineMSEFitness
from .bsplinefea import *

__all__ = [
    "__version__",
    "BSplinePSO",
    "BSplineGA",
    "BSplineGA",
    "BSplineMSEFitness",
]


# TODO: replace all imports above with this below?
# importlib.import_module("._versoin", "__version__")
