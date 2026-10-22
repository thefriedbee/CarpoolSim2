"""Tests for splitting a pool by trip similarity and stacking the pieces."""

import numpy as np
import pytest

from carpoolsim.trip_cluster import TripCluster
from carpoolsim.trip_cluster import combine_trips
from carpoolsim.trip_cluster import group_trips


def trips(*origins: tuple[float, float]) -> TripCluster:
    """One trip per origin, with the destination at the same point."""
    count = len(origins)
    ox = np.array([east for east, _north in origins], dtype=float)
    oy = np.array([north for _east, north in origins], dtype=float)
    return TripCluster(ox, oy, ox.copy(), oy.copy(), np.zeros(count), np.arange(count))


def test_a_small_pool_stays_one_cluster() -> None:
    cluster = trips((0.0, 0.0), (10.0, 0.0), (10_000.0, 0.0))

    groups = group_trips(cluster, max_size=3)

    assert len(groups) == 1
    assert groups[0].ids.tolist() == [0, 1, 2]
    assert groups[0].ox.tolist() == [0.0, 10.0, 10_000.0]


def test_identical_trips_split_in_row_order() -> None:
    cluster = trips(*[(0.0, 0.0)] * 5)

    groups = group_trips(cluster, max_size=2)
    combined = combine_trips(groups)

    assert [group.ids.tolist() for group in groups] == [[0, 1], [2], [3, 4]]
    assert [len(group) <= 2 for group in groups] == [True, True, True]
    assert combined.ids.tolist() == [0, 1, 2, 3, 4]
    assert combined.ox.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert group_trips(cluster, max_size=2)[0].ids.tolist() == groups[0].ids.tolist()


def test_separated_clouds_are_not_mixed() -> None:
    cluster = trips(
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (50_000.0, 0.0),
        (50_001.0, 0.0),
        (50_000.0, 1.0),
        (50_001.0, 1.0),
    )

    groups = group_trips(cluster, max_size=4)
    combined = combine_trips(groups)

    assert [group.ids.tolist() for group in groups] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert combined.ids.tolist() == list(range(8))
    assert combined.dx.tolist() == cluster.dx.tolist()


def test_groups_cover_every_traveler_once() -> None:
    origins = [(float(index * 1_000), float(index % 3)) for index in range(7)]
    cluster = trips(*origins)

    groups = group_trips(cluster, max_size=3)
    ids = [int(trip_id) for group in groups for trip_id in group.ids]

    assert sorted(ids) == list(range(7))
    assert len(ids) == len(set(ids))
    assert max(len(group) for group in groups) <= 3


def test_an_empty_pool_combines_to_an_empty_cluster() -> None:
    cluster = trips()

    assert group_trips(cluster, max_size=2) == []
    assert len(combine_trips([])) == 0


def test_combined_ids_must_be_unique() -> None:
    cluster = trips((0.0, 0.0))

    with pytest.raises(ValueError, match="ids must be unique"):
        combine_trips([cluster, cluster])


def test_max_size_must_be_an_integer_of_at_least_two() -> None:
    cluster = trips((0.0, 0.0))

    with pytest.raises(ValueError, match="max_size must be at least 2"):
        group_trips(cluster, max_size=1)
    with pytest.raises(ValueError, match="max_size must be an integer"):
        group_trips(cluster, max_size=2.5)
