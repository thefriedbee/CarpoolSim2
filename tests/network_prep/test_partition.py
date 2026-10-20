"""Tests for geographic network partition."""

import numpy as np
import pytest

from carpoolsim.network_prep.partition import partition_network
from tests.conftest import Network


def test_line_splits_on_east_west_cuts(line_network: Network) -> None:
    nodes, edges = line_network

    part = partition_network(nodes, edges, max_nodes=3, min_share=0.30)

    assert [members.tolist() for members in part.parts] == [[0, 1], [2, 3], [4, 5]]
    assert part.cell_of.tolist() == [0, 0, 1, 1, 2, 2]
    assert part.src.tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    assert part.dst.tolist() == [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
    assert set(part.portal_ids.tolist()) == {1, 2, 3, 4}
    assert part.portal_counts().tolist() == [1, 2, 1]
    assert part.n_links == 5
    assert part.n_boundary_links == 2


def test_parts_cover_each_node_once(split_network: Network) -> None:
    nodes, edges = split_network
    max_nodes = 3

    part = partition_network(nodes, edges, max_nodes=max_nodes, min_share=0.30)

    covered = np.concatenate(part.parts)
    assert sorted(int(node) for node in covered) == list(range(len(nodes)))
    assert part.n_parts == len(part.parts)
    assert int(part.part_sizes().max()) <= max_nodes
    assert part.part_sizes().tolist() == [len(members) for members in part.parts]
    assert int(part.portal_counts().sum()) == part.n_portals
    for cell_id, members in enumerate(part.parts):
        assert set(part.cell_of[members].tolist()) == {cell_id}

    crossing = set()
    undirected = set()
    for start, end in zip(part.src.tolist(), part.dst.tolist(), strict=True):
        if start == end:
            continue
        link = (min(start, end), max(start, end))
        undirected.add(link)
        if part.cell_of[start] != part.cell_of[end]:
            crossing.add(link)
    assert part.n_links == len(undirected)
    assert part.n_boundary_links == len(crossing)
    assert set(part.portal_ids.tolist()) == {node for link in crossing for node in link}
    portal_ids = set(part.portal_ids.tolist())
    assert part.is_portal.tolist() == [i in portal_ids for i in range(len(nodes))]


def test_small_limit_is_not_required_for_a_short_line(line_network: Network) -> None:
    nodes, edges = line_network

    part = partition_network(nodes, edges, max_nodes=100)

    assert part.n_parts == 1
    assert part.n_portals == 0
    assert part.n_boundary_links == 0
    assert part.parts[0].tolist() == list(range(len(nodes)))


@pytest.mark.parametrize("max_nodes", [0, 1])
def test_max_nodes_below_two_raises(line_network: Network, max_nodes: int) -> None:
    nodes, edges = line_network
    with pytest.raises(ValueError, match="max_nodes must be at least 2"):
        partition_network(nodes, edges, max_nodes=max_nodes)


@pytest.mark.parametrize("min_share", [0.0, 0.5, 1.0])
def test_min_share_outside_open_interval_raises(
    line_network: Network,
    min_share: float,
) -> None:
    nodes, edges = line_network
    with pytest.raises(ValueError, match="min_share must be between 0 and 0.5"):
        partition_network(nodes, edges, min_share=min_share)
