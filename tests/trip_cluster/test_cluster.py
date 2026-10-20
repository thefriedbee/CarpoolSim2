"""Tests for an ordered trip cluster."""

from pathlib import Path

import numpy as np
import pytest

from carpoolsim.trip_cluster import TripCluster

EXAMPLE = (
    Path(__file__).resolve().parents[2] / "data_inputs" / "trip" / "gt_2022_off_campus_trips.csv"
)


def test_arrays_keep_traveler_order() -> None:
    cluster = TripCluster(
        ox=np.array([0.0, 2.0]),
        oy=np.array([1.0, 3.0]),
        dx=np.array([4.0, 6.0]),
        dy=np.array([5.0, 7.0]),
        t=np.array([458.0, 466.0]),
    )

    assert len(cluster) == 2
    assert cluster.ox.shape == (2,)
    assert cluster.oy.tolist() == [1.0, 3.0]
    assert cluster.dx.tolist() == [4.0, 6.0]
    assert cluster.dy.tolist() == [5.0, 7.0]
    assert cluster.t.tolist() == [458.0, 466.0]
    assert cluster.ids.tolist() == [0, 1]


def test_explicit_ids_are_kept_and_duplicates_are_rejected() -> None:
    cluster = TripCluster(
        ox=np.zeros(2),
        oy=np.zeros(2),
        dx=np.zeros(2),
        dy=np.zeros(2),
        t=np.zeros(2),
        ids=np.array([10, 20]),
    )

    assert cluster.ids.tolist() == [10, 20]
    with pytest.raises(ValueError, match="ids must be unique"):
        TripCluster(
            ox=np.zeros(2),
            oy=np.zeros(2),
            dx=np.zeros(2),
            dy=np.zeros(2),
            t=np.zeros(2),
            ids=np.array([10, 10]),
        )
    with pytest.raises(ValueError, match="ids must be a flat array of the same length"):
        TripCluster(
            ox=np.zeros(2),
            oy=np.zeros(2),
            dx=np.zeros(2),
            dy=np.zeros(2),
            t=np.zeros(2),
            ids=np.array([10]),
        )


def test_csv_trip_id_becomes_the_traveler_id(tmp_path: Path) -> None:
    path = tmp_path / "trips.csv"
    path.write_text(
        "trip_id,origin_x,origin_y,dest_x,dest_y,departure_time\n"
        "10,-1,2,-3,4,07:38\n"
        "12,-5,6,-7,8,07:46\n",
        encoding="utf-8",
    )

    cluster = TripCluster.from_csv(path)

    assert cluster.ids.tolist() == [10, 12]


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        TripCluster(
            ox=np.zeros(2),
            oy=np.zeros(2),
            dx=np.zeros(2),
            dy=np.zeros(1),
            t=np.zeros(2),
        )


def test_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "trips.csv"
    path.write_text(
        "origin_x,origin_y,dest_x,dest_y,departure_time\n"
        "-9399966.2,3987110.7,-9394901.2,3998741.4,07:38\n"
        "-9397185.9,3991863.4,-9394901.2,3998741.4,07:46\n",
        encoding="utf-8",
    )

    cluster = TripCluster.from_csv(path)

    assert cluster.ox.tolist() == [-9399966.2, -9397185.9]
    assert cluster.oy[0] == 3987110.7
    assert cluster.dx.tolist() == [-9394901.2, -9394901.2]
    assert cluster.dy[1] == 3998741.4
    assert cluster.t.tolist() == [7 * 60 + 38, 7 * 60 + 46]
    assert cluster.ids.tolist() == [0, 1]


def test_example_file_loads_in_order() -> None:
    cluster = TripCluster.from_csv(EXAMPLE)

    assert len(cluster) == 1539
    assert cluster.ox.shape == (1539,)
    assert cluster.oy.shape == (1539,)
    assert cluster.dx.shape == (1539,)
    assert cluster.dy.shape == (1539,)
    assert cluster.t.shape == (1539,)
    assert cluster.ox[0] == -9399966.2
    assert cluster.oy[0] == 3987110.7
    assert cluster.t[0] == 7 * 60 + 38
    assert cluster.t[3] == 8 * 60 + 56
    assert cluster.ids[0] == 1
    assert cluster.ids[1] == 2
    assert len(set(cluster.ids.tolist())) == len(cluster)
