"""Choose conflict-free driver and rider pairs from a shareability matrix."""

from collections.abc import Callable

import numpy as np

from carpoolsim.optimizer.bipartite import BipartiteAssigner
from carpoolsim.optimizer.greedy import GreedyAssigner
from carpoolsim.optimizer.matching import MatchingAssigner

Assigner = Callable[[np.ndarray, np.ndarray], list[tuple[int, int]]]

__all__ = [
    "Assigner",
    "BipartiteAssigner",
    "GreedyAssigner",
    "MatchingAssigner",
]
