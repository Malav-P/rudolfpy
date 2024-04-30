"""Init for Kalman filter module, rudolfpy."""

# Let users know if they're missing any of our hard dependencies
hard_dependencies = (
    "numpy",
    "matplotlib",
    "numba",
    "scipy",
    "tqdm",
)
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies

# Imports
from .dynamics import *
from .measurement import *
from ._base_filter import BaseFilter, unbiased_random_process_3dof
from ._ekf import ExtendedKalmanFilter
from ._recurse_filter import Recursor