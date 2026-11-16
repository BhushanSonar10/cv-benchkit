# cv-benchkit v0.1.0 Community Launch Kit

This document contains copy-paste launch templates formatted for developer communities across Reddit, X (Twitter), and Discord. Each post emphasizes zero-dependency speed, bidirectional annotation conversion (COCO, YOLO, Pascal VOC), vectorized IoU/GIoU benchmarking, and our welcoming Good First Issues onboarding pipeline.

---

## 1. Reddit: r/computervision

**Title:** `[P] cv-benchkit: A lightweight Python library for fast CV dataset conversion and vectorized IoU benchmarking (v0.1.0 live on PyPI)`

**Flair:** Project

**Body:**

```markdown
Hey r/computervision,

Every time I start a new object detection project, I find myself rewriting the same conversion scripts between COCO JSON, YOLO txt, and Pascal VOC XML, or pulling in massive frameworks just to compute bounding box metrics like IoU, GIoU, DIoU, or confusion matrices.

To solve this friction, I built and just published **cv-benchkit** (v0.1.0) on PyPI:
- **PyPI:** https://pypi.org/project/cv-benchkit/
- **GitHub:** https://github.com/udayraj1238/cv-benchkit

### What it does:
1. **Lossless Dataset Conversions:** Bidirectional conversion between COCO (`instances.json`), YOLO (`<id>.txt` normalized), and Pascal VOC XML. Handles coordinate clamping, normalization/denormalization, and multi-category remapping automatically.
2. **Vectorized Metrics Engine:** Pure NumPy/PyTorch implementations of IoU, GIoU (Generalized IoU), DIoU (Distance IoU), and CIoU (Complete IoU), plus confusion matrices across class thresholds.
3. **CLI + Python API:** Batch convert datasets in one terminal command:
   ```bash
   pip install cv-benchkit
   cv-benchkit convert --src-format coco --dst-format yolo --src annotations.json --dst ./yolo_labels/
   ```
4. **Zero Overhead:** No heavy runtime requirements, tested across Python 3.10–3.13, with 100% test coverage and strict type annotations (`mypy --strict`).

### Quick Python Example:
```python
import numpy as np
from cv_benchkit.metrics import vectorized_iou, vectorized_giou

boxes1 = np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float32)
boxes2 = np.array([[15, 15, 55, 55], [200, 200, 250, 250]], dtype=np.float32)

# Fast pairwise IoU and GIoU matrix computation
iou_matrix = vectorized_iou(boxes1, boxes2)
giou_matrix = vectorized_giou(boxes1, boxes2)
print("IoU Matrix:\n", iou_matrix)
print("GIoU Matrix:\n", giou_matrix)
```

### Call for Contributors (Good First Issues):
We have curated **15 Good First Issues** tagged with reproducible instructions, expected outputs, and test scaffolds:
https://github.com/udayraj1238/cv-benchkit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22

If you want to contribute to an active CV open source project (e.g., adding Albumentations format support, benchmarking visualization, or precision-recall curves), we would love to review and merge your PRs! Feedback and feature requests are very welcome.
```

---

## 2. Reddit: r/Python

**Title:** `Showcase: cv-benchkit - Fast computer vision annotation converter and vectorized metrics toolkit (v0.1.0)`

**Flair:** Showcase

**Body:**

```markdown
Hi everyone,

I'm excited to share **cv-benchkit**, an open-source Python toolkit designed to streamline computer vision dataset preparation and evaluation pipelines without dragging in gigabytes of dependencies.

- **PyPI:** https://pypi.org/project/cv-benchkit/
- **GitHub:** https://github.com/udayraj1238/cv-benchkit
- **Documentation & Walkthrough:** [Quickstart Walkthrough](https://github.com/udayraj1238/cv-benchkit/blob/main/examples/quickstart_conversion_walkthrough.py)

### Why build this?
Most CV practitioners end up with ad-hoc scripts converting between COCO, YOLO, and Pascal VOC formats. These scripts often contain subtle coordinate rounding or off-by-one errors (0-indexed vs 1-indexed, corner vs center formats, normalized vs pixel coordinates). 

`cv-benchkit` provides a unified, strictly-typed data abstraction layer for datasets and bounding box geometry.

### Key Highlights:
- **Strict Typing & Modern Python:** Built for Python 3.10 through 3.13 with 100% type annotations verified under `mypy --strict`.
- **Fast Vectorized Operations:** Vectorized pairwise IoU, GIoU, DIoU, and CIoU computed with NumPy / PyTorch broadcasting.
- **CLI Utility:** `cv-benchkit convert` and `cv-benchkit evaluate` work out of the box for shell scripts and CI jobs.
- **High Test Coverage:** Comprehensive test suite covering coordinate edge cases (zero-area boxes, boundaries, out-of-frame annotations).

### Installation:
```bash
pip install cv-benchkit
# or using uv:
uv add cv-benchkit
```

### Open Source & Community:
`cv-benchkit` is governed under Apache-2.0 and actively maintained with automated CI, OpenSSF security practices, and Dependabot.

If you are looking to contribute to open-source Python, check out our **15 curated Good First Issues**:
https://github.com/udayraj1238/cv-benchkit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22

All feedback, bug reports, and pull requests are welcomed!
```

---

## 3. X (Twitter) Launch Thread

**Post 1 (Hook & PyPI Release):**
```
🚀 Excited to launch cv-benchkit v0.1.0 on PyPI!

A fast, lightweight Python toolkit for computer vision dataset conversion (COCO ⇄ YOLO ⇄ VOC) and vectorized bounding box metrics (IoU, GIoU, DIoU, CIoU).

📦 PyPI: https://pypi.org/project/cv-benchkit/
⭐ GitHub: https://github.com/udayraj1238/cv-benchkit

🧵 1/4
```

**Post 2 (Features & Architecture):**
```
Tired of rewriting annotation converters with subtle off-by-one coordinate bugs?

cv-benchkit handles:
✅ Bidirectional COCO, YOLO, and Pascal VOC transformations
✅ Vectorized pairwise box metrics via NumPy / PyTorch
✅ Python 3.10–3.13 support & strict @mypy typing
✅ Clean CLI for batch processing

🧵 2/4
```

**Post 3 (Quickstart Snippet):**
```
Get started in seconds:

pip install cv-benchkit

Vectorized metrics in 3 lines:
```python
from cv_benchkit.metrics import vectorized_iou

# Pairwise NxM matrix in milliseconds
iou = vectorized_iou(pred_boxes, gt_boxes)
```

Full walkthrough script included in the repo!
🧵 3/4
```

**Post 4 (Call for Contributors):**
```
Looking to contribute to open source? 🤝

We opened 15 beginner-friendly Good First Issues with complete step-by-step guidance and test scaffolds:
👉 https://github.com/udayraj1238/cv-benchkit/issues

Feedback and stars ⭐ are greatly appreciated!
🧵 4/4
```

---

## 4. Discord (PyTorch / MLOps / Computer Vision Servers)

**Channel:** `#projects-showcase` or `#open-source`

**Message:**
```markdown
👋 Hi everyone! I just released **cv-benchkit v0.1.0** on PyPI.

It’s an open-source toolkit designed to eliminate the boilerplate of dataset format wrangling and evaluation in computer vision workflows.

**Highlights:**
• **Lossless Annotation Conversions:** Convert seamlessly between COCO JSON, YOLO TXT, and Pascal VOC XML.
• **Vectorized Metrics:** Fast IoU, GIoU, DIoU, CIoU, and confusion matrix computation.
• **CLI + Python API:** Use directly in Python code or as a terminal utility (`cv-benchkit convert`).
• **Strict Quality:** Full mypy --strict compliance, 100% test coverage, Apache 2.0 license.

🔗 **Links:**
• PyPI: <https://pypi.org/project/cv-benchkit/>
• GitHub: <https://github.com/udayraj1238/cv-benchkit>

We also have **15 curated Good First Issues** if you want to contribute to an active Python CV project:
<https://github.com/udayraj1238/cv-benchkit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>

Would love to hear your thoughts, feedback, or feature requests!
```
