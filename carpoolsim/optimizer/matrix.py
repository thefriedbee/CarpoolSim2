"""Check that traveler ids line up with a shareability matrix."""

from __future__ import annotations

import numpy as np


def aligned_ids(matrix: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Return one integer id per row of a square ``matrix``.

    The result follows row order, so entry ``i`` is the traveler in that row.
    Ids are unique. A matrix cut out of a larger day still carries the ids
    from that day, which is how a pair can be traced after the rows move.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    labels = np.asarray(ids)
    if labels.ndim != 1 or labels.shape != (matrix.shape[0],):
        raise ValueError("ids must label each row of the matrix")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("ids must be integers")
    labels = labels.astype(np.int64, copy=False)
    if len(set(labels.tolist())) != labels.shape[0]:
        raise ValueError("ids must be unique")
    return labels
