"""
Quickstart Conversion & Evaluation Walkthrough for cv-benchkit.

Demonstrates:
1. Low-level vectorized coordinate transformations:
   xyxy <-> xywh <-> cxcywh <-> normalized YOLO
2. Pairwise metric calculations:
   IoU and Generalized IoU (GIoU)
3. Multiclass confusion matrix with background handling
4. End-to-end dataset schema translation:
   COCO JSON -> YOLO format -> Pascal VOC XML -> COCO JSON roundtrip
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from cv_benchkit import (
    COCOAnnotation,
    COCODataset,
    coco_to_voc,
    coco_to_yolo,
    compute_confusion_matrix,
    compute_giou,
    compute_iou,
    cxcywh_to_xyxy,
    normalized_to_xyxy,
    voc_to_coco,
    xywh_to_xyxy,
    xyxy_to_cxcywh,
    xyxy_to_normalized,
    xyxy_to_xywh,
    yolo_to_coco,
)
from cv_benchkit.converters import COCOCategory, COCOImage


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def main() -> None:
    section("1. Low-Level Vectorized Bounding Box Conversions")

    # Sample pixel coordinates [x1, y1, x2, y2]
    boxes_xyxy = np.array(
        [
            [50.0, 50.0, 200.0, 300.0],
            [100.0, 150.0, 400.0, 450.0],
        ]
    )
    img_w, img_h = 640, 480
    print(f"Original xyxy boxes:\n{boxes_xyxy}")

    # xyxy -> xywh (COCO) -> xyxy
    boxes_xywh = xyxy_to_xywh(boxes_xyxy)
    print(f"\nConverted to xywh (COCO [x, y, w, h]):\n{boxes_xywh}")
    recovered_xyxy = xywh_to_xyxy(boxes_xywh)
    assert np.allclose(boxes_xyxy, recovered_xyxy)

    # xyxy -> cxcywh (center format)
    boxes_cxcywh = xyxy_to_cxcywh(boxes_xyxy)
    print(f"\nConverted to cxcywh (center):\n{boxes_cxcywh}")
    recovered_from_center = cxcywh_to_xyxy(boxes_cxcywh)
    assert np.allclose(boxes_xyxy, recovered_from_center)

    # xyxy -> normalized YOLO [cx, cy, w, h] in [0, 1]
    yolo_norm = xyxy_to_normalized(boxes_xyxy, img_w=img_w, img_h=img_h)
    print(f"\nNormalized YOLO coordinates (img_size={img_w}x{img_h}):\n{yolo_norm}")
    denorm_xyxy = normalized_to_xyxy(yolo_norm, img_w=img_w, img_h=img_h)
    assert np.allclose(boxes_xyxy, denorm_xyxy)
    print("  -> All roundtrip coordinate conversions verified successfully.")

    section("2. Vectorized Metric Evaluations (IoU & GIoU)")

    gt_boxes = np.array(
        [
            [50.0, 50.0, 150.0, 150.0],
            [200.0, 200.0, 300.0, 300.0],
        ]
    )
    pred_boxes = np.array(
        [
            [60.0, 60.0, 160.0, 160.0],  # High overlap with box 0
            [400.0, 400.0, 500.0, 500.0],  # Disjoint from all boxes
        ]
    )

    ious = compute_iou(gt_boxes, pred_boxes)
    print(f"Ground Truth (2 boxes) vs Predictions (2 boxes) - IoU Matrix:\n{ious}")

    gious = compute_giou(gt_boxes, pred_boxes)
    print(f"\nGeneralized IoU (GIoU Matrix - negative penalty for disjoint boxes):\n{gious}")

    section("3. Multiclass Confusion Matrix with Background Class")

    # True boxes & labels
    true_b = np.array(
        [
            [50.0, 50.0, 150.0, 150.0],
            [200.0, 200.0, 300.0, 300.0],
        ]
    )
    true_l = np.array([0, 1])

    # Predicted boxes & labels (first matches class 0, second is a false positive)
    pred_b = np.array(
        [
            [55.0, 55.0, 145.0, 145.0],
            [500.0, 500.0, 600.0, 600.0],
        ]
    )
    pred_l = np.array([0, 1])

    cm = compute_confusion_matrix(
        gt_labels=true_l,
        pred_labels=pred_l,
        gt_boxes=true_b,
        pred_boxes=pred_b,
        num_classes=2,
        iou_threshold=0.5,
    )
    print(f"Confusion Matrix (shape {cm.shape}, index 2 = background):\n{cm}")

    section("4. End-to-End Dataset Schema Translation Walkthrough")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        yolo_dir = root / "yolo_labels"
        voc_dir = root / "voc_xml"

        # 1. Build an in-memory COCO dataset
        coco = COCODataset(
            images=[
                COCOImage(id=1, file_name="image_001.jpg", width=640, height=480),
                COCOImage(id=2, file_name="image_002.jpg", width=640, height=480),
            ],
            annotations=[
                COCOAnnotation(
                    id=1,
                    image_id=1,
                    category_id=0,
                    bbox=[50.0, 50.0, 100.0, 100.0],
                    area=10000.0,
                ),
                COCOAnnotation(
                    id=2,
                    image_id=2,
                    category_id=1,
                    bbox=[100.0, 100.0, 150.0, 200.0],
                    area=30000.0,
                ),
            ],
            categories=[
                COCOCategory(id=0, name="person"),
                COCOCategory(id=1, name="vehicle"),
            ],
        )
        print(f"Created COCO dataset: {len(coco.images)} images, {len(coco.annotations)} boxes.")

        # 2. Convert COCO -> YOLO
        yolo_files = coco_to_yolo(coco, yolo_dir)
        print(f"COCO -> YOLO: {len(yolo_files)} files written to {yolo_dir.name}/")
        for yf in yolo_files:
            print(f"  [{yf.name}]: {yf.read_text().strip()}")

        # 3. Convert YOLO -> COCO
        image_sizes = {"image_001": (640, 480), "image_002": (640, 480)}
        reloaded_coco = yolo_to_coco(
            yolo_dir,
            image_sizes=image_sizes,
            category_names=["person", "vehicle"],
        )
        print(f"YOLO -> COCO: {len(reloaded_coco.annotations)} annotations recovered.")

        # 4. Convert COCO -> Pascal VOC XML
        voc_files = coco_to_voc(coco, voc_dir)
        print(f"COCO -> VOC XML: {len(voc_files)} files written to {voc_dir.name}/")

        # 5. Convert Pascal VOC -> COCO
        voc_coco = voc_to_coco(voc_dir, category_names=["person", "vehicle"])
        print(f"VOC XML -> COCO: {len(voc_coco.annotations)} annotations parsed.")

    section("Walkthrough Complete! All conversions and metrics verified.")


if __name__ == "__main__":
    main()
