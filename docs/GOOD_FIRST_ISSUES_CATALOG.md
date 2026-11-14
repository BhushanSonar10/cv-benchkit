# Good First Issues Catalog: cv-benchkit

Welcome to the `cv-benchkit` community! Below is a catalog of 40 well-scoped, modular issues designed for first-time contributors. Each issue specifies the target file, requirements, acceptance criteria, and exact verification commands.

To claim an issue, comment on GitHub with:
`I'd like to work on this! Plan: [one sentence approach]`.

---

## Category 1: Documentation & Docstrings (Issues #1 – #10)

### Issue #1: Add NumPy-style docstrings with runnable doctests for `xyxy_to_xywh`
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Enhance `xyxy_to_xywh` docstring with clear parameter descriptions, return shapes, and a runnable doctest example.
- **Acceptance Criteria**: Running `uv run pytest --doctest-modules src/cv_benchkit/converters.py` passes without errors.
- **Verification**: `uv run pytest --doctest-modules src/cv_benchkit/converters.py`

### Issue #2: Add doctest example for `xywh_to_xyxy`
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Provide runnable docstring doctests for `xywh_to_xyxy` including edge-case verification.
- **Acceptance Criteria**: Doctests pass and `ruff check` reports no formatting issues.
- **Verification**: `uv run pytest --doctest-modules src/cv_benchkit/converters.py`

### Issue #3: Add docstring example with diagrams for `xyxy_to_cxcywh`
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Document the coordinate center formulas clearly in the docstring.
- **Acceptance Criteria**: Doctest and docstring completeness.
- **Verification**: `uv run pytest --doctest-modules src/cv_benchkit/converters.py`

### Issue #4: Add comprehensive docstring for `cxcywh_to_xyxy`
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Add parameter, return, and doctest blocks to `cxcywh_to_xyxy`.
- **Acceptance Criteria**: Complete parameter and return descriptions matching Sphinx/NumPy style.
- **Verification**: `uv run ruff check src/cv_benchkit/converters.py`

### Issue #5: Add visual ASCII diagram to `compute_iou` docstring
- **Target File**: `src/cv_benchkit/metrics.py`
- **Task**: Add an ASCII illustration of intersection over union in the docstring.
- **Acceptance Criteria**: Clean formatting, no line length > 100 characters.
- **Verification**: `uv run ruff check src/cv_benchkit/metrics.py`

### Issue #6: Document GIoU penalty term mathematically in `compute_giou`
- **Target File**: `src/cv_benchkit/metrics.py`
- **Task**: Detail the enclosing box formulation $C$ and penalty term $(|C \setminus (A \cup B)|) / |C|$.
- **Acceptance Criteria**: KaTeX/Markdown mathematical clarity, clean linting.
- **Verification**: `uv run ruff check src/cv_benchkit/metrics.py`

### Issue #7: Add docstring explanation of background index in `compute_confusion_matrix`
- **Target File**: `src/cv_benchkit/metrics.py`
- **Task**: Explain how the `num_classes + 1` row and column index maps to false positives and false negatives.
- **Acceptance Criteria**: Clear explanation with a mini $3 \times 3$ example table.
- **Verification**: `uv run ruff check src/cv_benchkit/metrics.py`

### Issue #8: Create `docs/cli_reference.md`
- **Target File**: `docs/cli_reference.md`
- **Task**: Document all `cv-benchkit` CLI options (`convert`, `validate`, flag descriptions).
- **Acceptance Criteria**: Accurate flags matching `src/cv_benchkit/cli.py`.
- **Verification**: `uv run cv-benchkit --help`

### Issue #9: Add quick-start tutorial notebook `examples/quickstart.ipynb`
- **Target File**: `examples/quickstart.ipynb`
- **Task**: Create an interactive Jupyter notebook demonstrating bbox conversion and IoU calculation.
- **Acceptance Criteria**: Executable end-to-end without errors.
- **Verification**: `uv run python -m jupyter nbconvert --to notebook --execute examples/quickstart.ipynb`

### Issue #10: Add type documentation table to `README.md`
- **Target File**: `README.md`
- **Task**: Add a Markdown table explaining supported coordinate formats (`xyxy`, `xywh`, `cxcywh`, `normalized`).
- **Acceptance Criteria**: Formatted table in README.
- **Verification**: Markdown rendering check.

---

## Category 2: Input Validation & Edge Cases (Issues #11 – #20)

### Issue #11: Validate non-negative bbox dimensions in `BBox`
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Add Pydantic validator ensuring width and height are strictly non-negative.
- **Acceptance Criteria**: Raises `ValueError` if $w < 0$ or $h < 0$.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #12: Add validation for non-inverted boxes in `xyxy_to_xywh`
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Check that $x_2 \ge x_1$ and $y_2 \ge y_1$; warn or fix inverted boxes with a parameter `sanitize=False`.
- **Acceptance Criteria**: `sanitize=True` swaps coordinates; `sanitize=False` raises `ValueError`.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #13: Support 1D single-box input in `xyxy_to_normalized`
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Ensure a 1D input shape `(4,)` returns shape `(1, 4)` consistently.
- **Acceptance Criteria**: Unit test asserting shape `(1, 4)` passes.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #14: Support 1D single-box input in `normalized_to_xyxy`
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Ensure a 1D input shape `(4,)` returns shape `(1, 4)`.
- **Acceptance Criteria**: Unit test in `tests/test_converters.py`.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #15: Validate image dimensions in `yolo_to_coco`
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Raise `ValueError` if any image stem in `image_sizes` has width $\le 0$ or height $\le 0$.
- **Acceptance Criteria**: Test verifies `ValueError` raised on zero or negative dimensions.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #16: Handle malformed line in YOLO text files gracefully
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: When reading YOLO text files, log a warning or raise a descriptive error if a line does not have 5 fields.
- **Acceptance Criteria**: Informative error message mentioning the file path and line number.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #17: Handle missing `<size>` tag in Pascal VOC XML
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Provide default fallback or log warning when `<size>` is missing in XML.
- **Acceptance Criteria**: Clean error or default handling without unhandled exceptions.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #18: Handle empty XML directory in `voc_to_coco`
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Return an empty `COCODataset` with valid empty lists when directory has no XML files.
- **Acceptance Criteria**: Returns `COCODataset(images=[], annotations=[], categories=[...])`.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #19: Add confidence score thresholding to `compute_confusion_matrix`
- **Target File**: `src/cv_benchkit/metrics.py`
- **Task**: Add optional parameter `pred_scores: NDArray | None = None` and `conf_threshold: float = 0.25`.
- **Acceptance Criteria**: Predictions below `conf_threshold` are filtered out before matching.
- **Verification**: `uv run pytest tests/test_metrics.py`

### Issue #20: Validate IoU threshold range in `compute_confusion_matrix`
- **Target File**: `src/cv_benchkit/metrics.py`
- **Task**: Raise `ValueError` if `iou_threshold` is not in $[0.0, 1.0]$.
- **Acceptance Criteria**: Test verifies exception when `iou_threshold = 1.5` or `-0.1`.
- **Verification**: `uv run pytest tests/test_metrics.py`

---

## Category 3: New Converters & Integrations (Issues #21 – #30)

### Issue #21: Implement `xywh_to_normalized` direct converter
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Implement direct conversion from COCO `[x, y, w, h]` to YOLO normalized `[cx, cy, nw, nh]`.
- **Acceptance Criteria**: Vectorized NumPy implementation with roundtrip test.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #22: Implement `normalized_to_xywh` direct converter
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Implement direct conversion from normalized YOLO to COCO `xywh`.
- **Acceptance Criteria**: Vectorized implementation with tests.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #23: Add COCO category filter utility
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Function `filter_coco_by_categories(coco: COCODataset, keep_categories: list[str]) -> COCODataset`.
- **Acceptance Criteria**: Filters annotations and categories while updating category IDs appropriately.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #24: Add COCO min-area filter utility
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Function `filter_coco_by_area(coco: COCODataset, min_area: float) -> COCODataset`.
- **Acceptance Criteria**: Removes annotations with area < `min_area`.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #25: Implement CSV format export for bounding boxes
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Export dataset annotations to standard CSV format: `filename,x1,y1,x2,y2,class_name`.
- **Acceptance Criteria**: CLI and Python API support.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #26: Implement CSV format import for bounding boxes
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Load standard CSV annotations into `COCODataset`.
- **Acceptance Criteria**: Vectorized parser with validation.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #27: Add batch box clipping utility `clip_boxes_to_image`
- **Target File**: `src/cv_benchkit/converters.py`
- **Task**: Clip xyxy boxes so coordinates stay within $[0, \text{width}]$ and $[0, \text{height}]$.
- **Acceptance Criteria**: Vectorized NumPy `np.clip` implementation with unit tests.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #28: Implement DIoU (Distance-IoU) metric
- **Target File**: `src/cv_benchkit/metrics.py`
- **Task**: Implement `compute_diou(boxes_a, boxes_b)` incorporating center distance penalty.
- **Acceptance Criteria**: Passes identity test (DIoU = 1 for identical boxes).
- **Verification**: `uv run pytest tests/test_metrics.py`

### Issue #29: Implement CIoU (Complete-IoU) metric
- **Target File**: `src/cv_benchkit/metrics.py`
- **Task**: Implement `compute_ciou(boxes_a, boxes_b)` incorporating aspect ratio penalty.
- **Acceptance Criteria**: Comprehensive unit tests covering various aspect ratios.
- **Verification**: `uv run pytest tests/test_metrics.py`

### Issue #30: Implement mean Average Precision (mAP@50) calculation
- **Target File**: `src/cv_benchkit/metrics.py`
- **Task**: Implement precision-recall curve computation and AP@50 for single class.
- **Acceptance Criteria**: Accurate AP calculation matching standard definitions.
- **Verification**: `uv run pytest tests/test_metrics.py`

---

## Category 4: Unit & Integration Tests (Issues #31 – #40)

### Issue #31: Add randomized hypothesis property tests for `xyxy_to_xywh`
- **Target File**: `tests/test_converters.py`
- **Task**: Use `hypothesis` to test coordinate roundtrip across 1,000 arbitrary float boxes.
- **Acceptance Criteria**: No roundtrip delta $> 10^{-5}$.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #32: Add randomized hypothesis tests for `xyxy_to_cxcywh`
- **Target File**: `tests/test_converters.py`
- **Task**: Property-based tests verifying roundtrip consistency.
- **Acceptance Criteria**: Passes with arbitrary positive float coordinates.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #33: Add extreme aspect ratio tests for `compute_giou`
- **Target File**: `tests/test_metrics.py`
- **Task**: Test GIoU behavior on extremely thin horizontal and vertical boxes ($1000 \times 1$ and $1 \times 1000$).
- **Acceptance Criteria**: Values remain bounded in $[-1.0, 1.0]$ without NaNs.
- **Verification**: `uv run pytest tests/test_metrics.py`

### Issue #34: Test identical prediction handling in confusion matrix
- **Target File**: `tests/test_metrics.py`
- **Task**: Verify confusion matrix when multiple predictions overlap a single ground truth.
- **Acceptance Criteria**: Only the highest scoring prediction matches; extra predictions count as false positives.
- **Verification**: `uv run pytest tests/test_metrics.py`

### Issue #35: Test CLI `--categories` argument parsing with spaces
- **Target File**: `tests/test_cli.py`
- **Task**: Verify category names containing spaces (e.g., `"traffic light"`, `"fire hydrant"`).
- **Acceptance Criteria**: CLI parses multi-word category names properly.
- **Verification**: `uv run pytest tests/test_cli.py`

### Issue #36: Test CLI invalid argument error codes
- **Target File**: `tests/test_cli.py`
- **Task**: Verify exit code 2 on unknown flags (e.g., `cv-benchkit --unknown`).
- **Acceptance Criteria**: Systematic check of error exit codes.
- **Verification**: `uv run pytest tests/test_cli.py`

### Issue #37: Benchmark IoU matrix speed against large batches
- **Target File**: `tests/test_benchmarks.py`
- **Task**: Add benchmark test verifying `compute_iou` on $1000 \times 1000$ boxes in under 50ms.
- **Acceptance Criteria**: Passes on standard hardware.
- **Verification**: `uv run pytest tests/test_benchmarks.py`

### Issue #38: Add end-to-end integration test: COCO -> YOLO -> VOC -> COCO
- **Target File**: `tests/test_converters.py`
- **Task**: Full 3-way cycle test asserting annotation preservation.
- **Acceptance Criteria**: Dataset survives full cycle with coordinates preserved within 1 pixel.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #39: Add test for empty annotation list in COCO JSON
- **Target File**: `tests/test_converters.py`
- **Task**: Verify handling of COCO files where `"annotations": []`.
- **Acceptance Criteria**: `coco_to_yolo` creates empty `.txt` files properly.
- **Verification**: `uv run pytest tests/test_converters.py`

### Issue #40: Add test for dataset with unannotated images
- **Target File**: `tests/test_converters.py`
- **Task**: Verify images present in `"images"` but with zero entries in `"annotations"` (background images).
- **Acceptance Criteria**: Empty label files created and image counts preserved.
- **Verification**: `uv run pytest tests/test_converters.py`
