"""Unit tests for compute_precision_recall_curve."""

from __future__ import annotations

import numpy as np
import pytest

from cv_benchkit import compute_precision_recall_curve


def test_perfect_detections() -> None:
    scores = np.array([0.9, 0.8, 0.7])
    matches = np.array([True, True, True])

    precision, recall = compute_precision_recall_curve(scores, matches, num_positives=3)

    np.testing.assert_allclose(precision, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(recall, [1 / 3, 2 / 3, 1.0])


def test_all_false_positives() -> None:
    scores = np.array([0.95, 0.5, 0.2])
    matches = np.array([False, False, False])

    precision, recall = compute_precision_recall_curve(scores, matches, num_positives=2)

    np.testing.assert_allclose(precision, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(recall, [0.0, 0.0, 0.0])


def test_mixed_scores_sorts_descending() -> None:
    # Unsorted scores: sorted order is 0.9 (True), 0.7 (False), 0.6 (True), 0.3 (False)
    scores = np.array([0.6, 0.9, 0.3, 0.7])
    matches = np.array([True, True, False, False])

    precision, recall = compute_precision_recall_curve(scores, matches, num_positives=2)

    np.testing.assert_allclose(precision, [1.0, 0.5, 2 / 3, 0.5])
    np.testing.assert_allclose(recall, [0.5, 0.5, 1.0, 1.0])


def test_single_detection() -> None:
    tp_prec, tp_rec = compute_precision_recall_curve(
        np.array([0.8]), np.array([True]), num_positives=1
    )
    np.testing.assert_allclose(tp_prec, [1.0])
    np.testing.assert_allclose(tp_rec, [1.0])

    fp_prec, fp_rec = compute_precision_recall_curve(
        np.array([0.8]), np.array([False]), num_positives=1
    )
    np.testing.assert_allclose(fp_prec, [0.0])
    np.testing.assert_allclose(fp_rec, [0.0])


def test_empty_inputs_and_zero_positives() -> None:
    prec_empty, rec_empty = compute_precision_recall_curve(
        np.empty(0, dtype=np.float64), np.empty(0, dtype=bool), num_positives=3
    )
    assert prec_empty.size == 0
    assert rec_empty.size == 0

    prec_zero_pos, rec_zero_pos = compute_precision_recall_curve(
        np.array([0.9, 0.4]), np.array([False, False]), num_positives=0
    )
    assert prec_zero_pos.size == 0
    assert rec_zero_pos.size == 0


def test_invalid_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_precision_recall_curve(
            np.array([0.9, 0.8]), np.array([True]), num_positives=1
        )

    with pytest.raises(ValueError, match="non-negative"):
        compute_precision_recall_curve(
            np.array([0.9]), np.array([True]), num_positives=-1
        )

    with pytest.raises(ValueError, match="1D"):
        compute_precision_recall_curve(
            np.array([[0.9, 0.8]]), np.array([True, False]), num_positives=1
        )
