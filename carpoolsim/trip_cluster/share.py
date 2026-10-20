"""Hold the candidate pairs for one trip cluster and apply filters to them."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from carpoolsim.trip_cluster.cluster import TripCluster

Filter = Callable[[TripCluster, np.ndarray], None]


class Shareability:
    """Candidate pairs among the travelers in one cluster.

    ``matrix[i, j]`` is true when traveler ``cluster.ids[i]`` can drive and
    traveler ``cluster.ids[j]`` can ride. Row ``i`` is the same traveler as
    ``cluster.t[i]``. The two directions can differ. The diagonal is false,
    because a trip is not a partner of itself.
    Each filter may turn pairs off and leaves pairs that are already off
    unchanged.
    """

    def __init__(self, cluster: TripCluster) -> None:
        self.cluster = cluster
        n = len(cluster)
        self.matrix = np.ones((n, n), dtype=bool)
        np.fill_diagonal(self.matrix, False)

    def apply(self, *filters: Filter) -> None:
        """Run each filter in order against this cluster and matrix."""
        for step in filters:
            step(self.cluster, self.matrix)
