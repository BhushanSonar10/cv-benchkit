"""Unit tests for compute_diou and compute_ciou."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cv_benchkit import compute_ciou, compute_diou
from cv_benchkit.metrics import compute_iou


def _reference(a: list[float], b: list[float]) -> tuple[float, float]:
    """Per-pair DIoU and CIoU written straight from the paper's formulas."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    wa, ha, wb, hb = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
    union = wa * ha + wb * hb - inter
    iou = inter / union if union > 0 else 0.0
    rho2 = ((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) ** 2 + (
        (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2
    ) ** 2
    c2 = (max(a[2], b[2]) - min(a[0], b[0])) ** 2 + (max(a[3], b[3]) - min(a[1], b[1])) ** 2
    diou = iou - rho2 / c2 if c2 > 0 else iou
    v = 4 / math.pi**2 * (math.atan2(wb, hb) - math.atan2(wa, ha)) ** 2
    denom = (1 - iou) + v
    alpha = v / denom if v > 0 and denom > 0 else 0.0
    return diou, diou - alpha * v


class TestDIoU:
    def test_identical_boxes_are_one(self) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 25.0, 15.0]])
        np.testing.assert_allclose(np.diag(compute_diou(boxes, boxes)), [1.0, 1.0])

    def test_disjoint_boxes(self) -> None:
        # IoU 0; centers (5, 5) and (25, 5); enclosing box 30 x 10.
        result = compute_diou(
            np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([[20.0, 0.0, 30.0, 10.0]])
        )
        np.testing.assert_allclose(result, [[0.0 - 400.0 / 1000.0]])

    def test_enclosing_boxes_with_same_center(self) -> None:
        # Same center, so the distance term is 0 and DIoU equals IoU.
        outer = np.array([[0.0, 0.0, 10.0, 10.0]])
        inner = np.array([[2.5, 2.5, 7.5, 7.5]])
        np.testing.assert_allclose(compute_diou(outer, inner), compute_iou(outer, inner))
        np.testing.assert_allclose(compute_diou(outer, inner), [[0.25]])

    def test_enclosing_boxes_off_center(self) -> None:
        outer = np.array([[0.0, 0.0, 10.0, 10.0]])
        inner = np.array([[0.0, 0.0, 5.0, 5.0]])
        # IoU 0.25; rho^2 = 2.5^2 + 2.5^2 = 12.5; c^2 = 200.
        np.testing.assert_allclose(compute_diou(outer, inner), [[0.25 - 12.5 / 200.0]])

    def test_zero_size_boxes_at_same_point_fall_back_to_iou(self) -> None:
        point = np.array([[3.0, 3.0, 3.0, 3.0]])
        np.testing.assert_allclose(compute_diou(point, point), [[0.0]])

    def test_range(self) -> None:
        rng = np.random.default_rng(0)
        xy = rng.uniform(0, 100, size=(40, 2))
        wh = rng.uniform(0, 50, size=(40, 2))
        boxes = np.hstack([xy, xy + wh])
        result = compute_diou(boxes[:20], boxes[20:])
        assert np.all(result > -1.0)
        assert np.all(result <= 1.0)


class TestCIoU:
    def test_identical_boxes_are_one(self) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 25.0, 15.0]])
        np.testing.assert_allclose(np.diag(compute_ciou(boxes, boxes)), [1.0, 1.0])

    def test_same_aspect_ratio_equals_diou(self) -> None:
        a = np.array([[0.0, 0.0, 10.0, 10.0]])
        b = np.array([[20.0, 0.0, 30.0, 10.0]])
        np.testing.assert_allclose(compute_ciou(a, b), compute_diou(a, b))

    def test_aspect_ratio_penalty(self) -> None:
        a = np.array([[0.0, 0.0, 10.0, 10.0]])
        b = np.array([[0.0, 0.0, 20.0, 10.0]])
        diou, ciou = _reference(a[0].tolist(), b[0].tolist())
        np.testing.assert_allclose(compute_diou(a, b), [[diou]])
        np.testing.assert_allclose(compute_ciou(a, b), [[ciou]])
        assert ciou < diou

    def test_can_fall_below_minus_one(self) -> None:
        a = np.array([[0.0, 0.0, 1.0, 100.0]])
        b = np.array([[1000.0, 0.0, 1100.0, 1.0]])
        assert compute_ciou(a, b)[0, 0] < -1.0
        assert compute_ciou(a, b)[0, 0] >= -1.5

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(1)
        xy = rng.uniform(0, 100, size=(10, 2))
        boxes = np.hstack([xy, xy + rng.uniform(0, 50, size=(10, 2))])
        np.testing.assert_allclose(
            compute_ciou(boxes[:4], boxes[4:]), compute_ciou(boxes[4:], boxes[:4]).T
        )

    def test_zero_height_box_does_not_divide_by_zero(self) -> None:
        line = np.array([[0.0, 5.0, 10.0, 5.0]])
        square = np.array([[0.0, 0.0, 10.0, 10.0]])
        with np.errstate(all="raise"):
            result = compute_ciou(line, square)
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result, [[_reference(line[0].tolist(), square[0].tolist())[1]]])


class TestAgainstReference:
    def test_random_pairs_match_scalar_reference(self) -> None:
        rng = np.random.default_rng(42)
        xy = rng.uniform(0, 100, size=(30, 2))
        wh = rng.uniform(0, 40, size=(30, 2))
        boxes = np.hstack([xy, xy + wh])
        a, b = boxes[:12], boxes[12:]
        diou = compute_diou(a, b)
        ciou = compute_ciou(a, b)
        assert diou.shape == ciou.shape == (12, 18)
        for i in range(12):
            for j in range(18):
                ref_diou, ref_ciou = _reference(a[i].tolist(), b[j].tolist())
                assert diou[i, j] == pytest.approx(ref_diou, abs=1e-12)
                assert ciou[i, j] == pytest.approx(ref_ciou, abs=1e-12)


@pytest.mark.parametrize("fn", [compute_diou, compute_ciou])
class TestShapesAndValidation:
    def test_empty_inputs(self, fn) -> None:  # type: ignore[no-untyped-def]
        boxes = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]])
        empty = np.empty((0, 4))
        assert fn(empty, boxes).shape == (0, 2)
        assert fn(boxes, empty).shape == (2, 0)
        assert fn(np.array([]), boxes).shape == (0, 2)

    def test_1d_input(self, fn) -> None:  # type: ignore[no-untyped-def]
        box = np.array([0.0, 0.0, 10.0, 10.0])
        assert fn(box, box).shape == (1, 1)

    @pytest.mark.parametrize(
        "bad",
        [
            [[0.0, 0.0, 1.0]],
            [[5.0, 0.0, 1.0, 1.0]],
            [[0.0, 0.0, np.nan, 1.0]],
            [[0.0, 0.0, np.inf, 1.0]],
        ],
    )
    def test_invalid_boxes_raise(self, fn, bad) -> None:  # type: ignore[no-untyped-def]
        good = np.array([[0.0, 0.0, 1.0, 1.0]])
        with pytest.raises(ValueError):
            fn(np.array(bad), good)
        with pytest.raises(ValueError):
            fn(good, np.array(bad))
