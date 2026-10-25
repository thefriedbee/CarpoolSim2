"""Float32 lengths for the Euclidean pair checks.

Trip coordinates, lot coordinates, departure times, and network travel times
stay float64. The checks cast the coordinates they read, then build distances
and ratios in float32, so a trips × trips × lots entry is four bytes.
"""

from __future__ import annotations

import numpy as np


def as_float32(values: np.ndarray) -> np.ndarray:
    """Return ``values`` as float32 before a distance is broadcast."""
    return np.asarray(values, dtype=np.float32)


def infinities(shape: tuple[int, ...]) -> np.ndarray:
    """Return ``shape`` filled with float32 infinity."""
    return np.full(shape, np.inf, dtype=np.float32)
