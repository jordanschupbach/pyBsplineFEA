# __name__ = "benchmarks"
# __name__ = "known_knots"

from .benchmarks import *

# from .known_knots_fea import KnownKnotsFea

__all__ = [
    "mdoppler",
    "big_spike",
    "cliff",
    "discontinuity",
    "smooth_peak",
    "second_smooth_peak",
]
