"""Evaluation metrics: IoU, GIoU, and confusion matrix computation.

All functions are vectorized with NumPy for efficient batch processing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def compute_iou(
    boxes_a: NDArray[np.floating[Any]],
    boxes_b: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Compute the pairwise Intersection-over-Union (IoU) matrix.

    Parameters
    ----------
    boxes_a : ndarray, shape ``(M, 4)``
        First set of boxes in xyxy format.
    boxes_b : ndarray, shape ``(N, 4)``
        Second set of boxes in xyxy format.

    Returns
    -------
    ndarray, shape ``(M, N)``
        IoU values in ``[0, 1]``.
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)

    if boxes_a.ndim == 1:
        boxes_a = boxes_a.reshape(1, 4)
    if boxes_b.ndim == 1:
        boxes_b = boxes_b.reshape(1, 4)

    m = boxes_a.shape[0]
    n = boxes_b.shape[0]
    if m == 0 or n == 0:
        return np.empty((m, n), dtype=np.float64)

    # Intersection
    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    # Areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection

    # Avoid division by zero
    iou: NDArray[np.floating[Any]] = np.where(union > 0, intersection / union, 0.0)
    return iou


def compute_giou(
    boxes_a: NDArray[np.floating[Any]],
    boxes_b: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Compute the pairwise Generalized IoU (GIoU) matrix.

    GIoU = IoU - |C \\ (A U B)| / |C|

    where C is the smallest enclosing box of A and B.

    Parameters
    ----------
    boxes_a : ndarray, shape ``(M, 4)``
        First set of boxes in xyxy format.
    boxes_b : ndarray, shape ``(N, 4)``
        Second set of boxes in xyxy format.

    Returns
    -------
    ndarray, shape ``(M, N)``
        GIoU values in ``[-1, 1]``.
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)

    if boxes_a.ndim == 1:
        boxes_a = boxes_a.reshape(1, 4)
    if boxes_b.ndim == 1:
        boxes_b = boxes_b.reshape(1, 4)

    m = boxes_a.shape[0]
    n = boxes_b.shape[0]
    if m == 0 or n == 0:
        return np.empty((m, n), dtype=np.float64)

    # Intersection
    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    # Areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection

    iou = np.where(union > 0, intersection / union, 0.0)

    # Enclosing box
    enc_x1 = np.minimum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    enc_y1 = np.minimum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    enc_x2 = np.maximum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    enc_y2 = np.maximum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    giou: NDArray[np.floating[Any]] = np.where(
        enc_area > 0,
        iou - (enc_area - union) / enc_area,
        iou,
    )
    return giou


def compute_confusion_matrix(
    gt_labels: NDArray[np.integer[Any]],
    pred_labels: NDArray[np.integer[Any]],
    gt_boxes: NDArray[np.floating[Any]],
    pred_boxes: NDArray[np.floating[Any]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> NDArray[np.int64]:
    """Compute an object-detection confusion matrix.

    Predictions are matched to ground-truth boxes using IoU. The matrix has
    shape ``(num_classes + 1, num_classes + 1)`` where the last row/column
    represents the "background" class (unmatched GT = FN, unmatched pred = FP).

    Parameters
    ----------
    gt_labels : ndarray, shape ``(G,)``
        Ground-truth class labels (0-indexed).
    pred_labels : ndarray, shape ``(P,)``
        Predicted class labels (0-indexed).
    gt_boxes : ndarray, shape ``(G, 4)``
        Ground-truth boxes in xyxy format.
    pred_boxes : ndarray, shape ``(P, 4)``
        Predicted boxes in xyxy format.
    num_classes : int
        Number of object classes (not counting background).
    iou_threshold : float
        IoU threshold for a match to count as a true positive. Must be in ``[0, 1]``.

    Returns
    -------
    ndarray, shape ``(num_classes + 1, num_classes + 1)``
        Confusion matrix where ``[i, j]`` = count of GT class *i* predicted as
        class *j*. The last index is "background".

    Raises
    ------
    ValueError
        If ``iou_threshold`` is outside ``[0, 1]``.
    """
    if not (0.0 <= iou_threshold <= 1.0):
        msg = f"iou_threshold must be between 0.0 and 1.0, got {iou_threshold}"
        raise ValueError(msg)

    gt_labels = np.asarray(gt_labels, dtype=np.int64)
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)

    if num_classes <= 0:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes}")
    if len(gt_labels) != len(gt_boxes):
        raise ValueError("Number of ground-truth labels must match number of ground-truth boxes")
    if len(pred_labels) != len(pred_boxes):
        raise ValueError("Number of predicted labels must match number of predicted boxes")

    bg = num_classes  # background index
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    num_gt = gt_labels.shape[0]
    num_pred = pred_labels.shape[0]

    if num_gt == 0 and num_pred == 0:
        return cm

    if num_gt == 0:
        # All predictions are false positives
        for pl in pred_labels:
            cm[bg, int(pl)] += 1
        return cm

    if num_pred == 0:
        # All ground truths are false negatives
        for gl in gt_labels:
            cm[int(gl), bg] += 1
        return cm

    iou_matrix = compute_iou(gt_boxes, pred_boxes)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    # Greedy matching: for each pred, find best IoU GT
    for pred_idx in range(num_pred):
        ious = iou_matrix[:, pred_idx]
        best_gt_idx = int(np.argmax(ious))
        if ious[best_gt_idx] >= iou_threshold and best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
            gt_cls = int(gt_labels[best_gt_idx])
            pred_cls = int(pred_labels[pred_idx])
            cm[gt_cls, pred_cls] += 1
        else:
            # False positive
            pred_cls = int(pred_labels[pred_idx])
            cm[bg, pred_cls] += 1

    # Unmatched GT → false negatives
    for gt_idx in range(num_gt):
        if gt_idx not in matched_gt:
            gt_cls = int(gt_labels[gt_idx])
            cm[gt_cls, bg] += 1

    return cm
