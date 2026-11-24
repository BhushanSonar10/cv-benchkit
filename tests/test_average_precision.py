"""Unit tests for compute_average_precision."""

from __future__ import annotations

import numpy as np
import pytest

from cv_benchkit import compute_average_precision


def test_perfect_detector_ap_is_one() -> None:
    precision = np.array([1.0, 1.0, 1.0])
    recall = np.array([1 / 3, 2 / 3, 1.0])

    assert compute_average_precision(precision, recall, method="all_points") == pytest.approx(1.0)
    assert compute_average_precision(precision, recall, method="interp_11") == pytest.approx(1.0)


def test_step_curve_analytical_all_points_vs_interp_11() -> None:
    # Step PR curve: (rec=0.2, prec=1.0), (rec=0.5, prec=0.5), (rec=0.8, prec=0.4)
    precision = np.array([1.0, 0.5, 0.4])
    recall = np.array([0.2, 0.5, 0.8])

    # all_points: 0.2*1.0 + 0.3*0.5 + 0.3*0.4 = 0.2 + 0.15 + 0.12 = 0.47
    ap_all = compute_average_precision(precision, recall, method="all_points")
    assert ap_all == pytest.approx(0.47)

    # interp_11: thresholds 0.0,0.1,0.2 -> 1.0 (3); 0.3,0.4,0.5 -> 0.5 (3);
    # 0.6,0.7,0.8 -> 0.4 (3); 0.9,1.0 -> 0.0 (2) => (3.0 + 1.5 + 1.2) / 11
    ap_11 = compute_average_precision(precision, recall, method="interp_11")
    assert ap_11 == pytest.approx((3.0 + 1.5 + 1.2) / 11.0)


def test_non_monotonic_precision_uses_envelope() -> None:
    # Precision rises from 0.5 to 0.8 at recall=0.5, so envelope at recall=0.25 is 0.8
    precision = np.array([0.5, 0.8])
    recall = np.array([0.25, 0.5])

    ap = compute_average_precision(precision, recall, method="all_points")
    assert ap == pytest.approx(0.5 * 0.8)


def test_empty_arrays_return_zero() -> None:
    empty = np.empty(0, dtype=np.float64)
    assert compute_average_precision(empty, empty, method="all_points") == 0.0
    assert compute_average_precision(empty, empty, method="interp_11") == 0.0


def test_invalid_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported AP method"):
        compute_average_precision(np.array([1.0]), np.array([1.0]), method="invalid")

    with pytest.raises(ValueError, match="1D"):
        compute_average_precision(np.array([[1.0]]), np.array([1.0]))

    with pytest.raises(ValueError, match="same shape"):
        compute_average_precision(np.array([1.0, 0.5]), np.array([0.5]))

    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        compute_average_precision(np.array([1.2]), np.array([0.5]))

    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        compute_average_precision(np.array([0.5]), np.array([-0.1]))

    with pytest.raises(ValueError, match="finite"):
        compute_average_precision(np.array([np.nan]), np.array([0.5]))
