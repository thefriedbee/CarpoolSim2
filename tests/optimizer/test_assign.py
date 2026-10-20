"""Tests for conflict-free assignment of a shareability matrix."""

import numpy as np
import pytest

from carpoolsim.optimizer import BipartiteAssigner
from carpoolsim.optimizer import GreedyAssigner
from carpoolsim.optimizer import MatchingAssigner


def blocking_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Four travelers where the first true cell blocks two disjoint pairs.

    Traveler 10 can drive 20. Traveler 20 can drive 40. Traveler 30 can
    drive 10. A scan that takes (10, 20) first leaves 30 and 40 unmatched.
    The two other cells share no traveler.
    """
    matrix = np.zeros((4, 4), dtype=bool)
    matrix[0, 1] = True
    matrix[1, 3] = True
    matrix[2, 0] = True
    return matrix, np.array([10, 20, 30, 40])


def test_greedy_assigner_uses_each_traveler_once_in_row_major_order() -> None:
    matrix = np.array(
        [
            [False, True, True],
            [True, False, True],
            [True, True, False],
        ]
    )
    skipped = np.array(
        [
            [False, False, True],
            [False, False, True],
            [True, True, False],
        ]
    )
    ids = np.arange(3)

    assert GreedyAssigner()(matrix, ids) == [(0, 1)]
    assert GreedyAssigner()(skipped, ids) == [(0, 2)]


def test_greedy_assigner_returns_traveler_ids() -> None:
    matrix = np.array(
        [
            [False, True],
            [False, False],
        ]
    )

    assert GreedyAssigner()(matrix, np.array([10, 20])) == [(10, 20)]


def test_matching_keeps_the_two_pairs_the_first_cell_blocks() -> None:
    matrix, ids = blocking_matrix()

    assert GreedyAssigner()(matrix, ids) == [(10, 20)]
    assert BipartiteAssigner()(matrix, ids) == [(10, 20)]
    assert MatchingAssigner()(matrix, ids) == [(20, 40), (30, 10)]


def test_both_directions_choose_one_driver() -> None:
    matrix = np.array(
        [
            [False, True],
            [True, False],
        ]
    )
    ids = np.array([10, 4])

    assert GreedyAssigner()(matrix, ids) == [(10, 4)]
    assert BipartiteAssigner()(matrix, ids) == [(10, 4)]
    assert MatchingAssigner()(matrix, ids) == [(4, 10)]


def test_an_empty_matrix_assigns_nobody() -> None:
    matrix = np.zeros((0, 0), dtype=bool)
    ids = np.zeros(0, dtype=np.int64)

    assert GreedyAssigner()(matrix, ids) == []
    assert BipartiteAssigner()(matrix, ids) == []
    assert MatchingAssigner()(matrix, ids) == []


def test_ids_must_label_each_row_of_a_square_matrix() -> None:
    matrix = np.zeros((2, 2), dtype=bool)

    with pytest.raises(ValueError, match="ids must label each row"):
        GreedyAssigner()(matrix, np.array([1]))
    with pytest.raises(ValueError, match="ids must be unique"):
        GreedyAssigner()(matrix, np.array([1, 1]))
    with pytest.raises(ValueError, match="matrix must be square"):
        GreedyAssigner()(np.zeros((2, 1), dtype=bool), np.array([1, 2]))
