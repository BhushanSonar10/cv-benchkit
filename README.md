# cv-benchkit

[![CI](https://github.com/udayraj1238/cv-benchkit/actions/workflows/ci.yml/badge.svg)](https://github.com/udayraj1238/cv-benchkit/actions)
[![PyPI](https://img.shields.io/pypi/v/cv-benchkit.svg)](https://pypi.org/project/cv-benchkit/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Good First Issues](https://img.shields.io/github/issues/udayraj1238/cv-benchkit/good%20first%20issue?label=good%20first%20issues&color=7057ff)](https://github.com/udayraj1238/cv-benchkit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

Unified computer vision annotation converter and model benchmarking toolkit for PyTorch and NumPy.

---

## Quickstart & Installation

Install `cv-benchkit` via `pip`:
```bash
pip install cv-benchkit
```

Or using `uv`:
```bash
uv add cv-benchkit
```

Or run directly without installation via `uvx`:
```bash
uvx --from cv-benchkit cv-benchkit --help
```

---

## Features

- **Format Converters**: Seamless, vectorized conversions between `xyxy`, `xywh` (COCO), `cxcywh`, and normalized (YOLO) bounding-box representations.
- **Dataset Translators**: Two-way translations between **COCO JSON**, **YOLO txt**, and **Pascal VOC XML**.
- **Vectorized Evaluation Metrics**: Fast NumPy implementations of pairwise **IoU**, **GIoU**, and multiclass **confusion matrix** calculations with background handling.
- **CLI Utility**: Command-line interface for fast dataset conversion and validation.
- **Strictly Typed & Tested**: 100% typed with PEP 561 marker, passing `mypy --strict`, with 95%+ unit test coverage.

---

## Python API Usage

```python
import numpy as np
from cv_benchkit import compute_iou, xyxy_to_normalized, validate_coco_file

# Compute pairwise IoU
boxes_a = np.array([[0, 0, 100, 100], [50, 50, 150, 150]])
boxes_b = np.array([[20, 20, 120, 120]])
iou_matrix = compute_iou(boxes_a, boxes_b)
print(iou_matrix)

# Validate and parse COCO annotations
dataset = validate_coco_file("path/to/annotations.json")
```

---

## CLI Usage

```bash
# Validate COCO dataset
cv-benchkit validate annotations.json

# Convert COCO to YOLO format
cv-benchkit convert --from coco --to yolo --input annotations.json --output ./yolo_labels

# Convert YOLO to COCO format
cv-benchkit convert --from yolo --to coco --input ./yolo_labels --output ./coco_out --image-width 640 --image-height 480 --categories person car bicycle
```

---

## Contributing & Good First Issues

We welcome first-time open-source contributors! 🚀

We have curated and actively maintain **modular, beginner-friendly issues** with clear acceptance criteria and verification commands:

👉 **[Browse All Open Good First Issues (15+ Available)](https://github.com/udayraj1238/cv-benchkit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)**

### How to Claim an Issue:
1. Find an issue labeled [`good first issue`](https://github.com/udayraj1238/cv-benchkit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) that is not yet assigned.
2. Comment on the issue: `I'd like to work on this! Plan: [one sentence approach]`.
3. Read [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, testing, and pull request guidelines.
4. Also see our [Good First Issues Catalog](docs/GOOD_FIRST_ISSUES_CATALOG.md) for a comprehensive catalog of upcoming tasks across documentation, converters, and metrics.

### Local Development Setup:
```bash
git clone https://github.com/udayraj1238/cv-benchkit.git
cd cv-benchkit
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run mypy src/
```

---

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
