"""Regression coverage for BBox dimension validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cv_benchkit.converters import BBox


@pytest.mark.parametrize("fmt", ["xyxy", "xywh", "cxcywh", "normalized"])
@pytest.mark.parametrize(
    ("width", "height", "valid"),
    [
        (-0.1, 0.2, False),
        (0.2, -0.1, False),
        (-0.1, -0.1, False),
        (0.0, 0.0, True),
        (0.0, 0.2, True),
        (0.2, 0.0, True),
        (0.1, 0.2, True),
    ],
)
def test_bbox_requires_non_negative_dimensions(
    fmt: str, width: float, height: float, valid: bool
) -> None:
    coords = [-0.5, -0.4, width, height]
    if fmt == "xyxy":
        coords = [-0.5, -0.4, -0.5 + width, -0.4 + height]

    error = None
    box = None
    try:
        box = BBox(fmt=fmt, coords=coords)
    except ValidationError as exc:
        error = exc

    if valid:
        assert error is None
        assert box is not None
        assert box.coords == coords
    else:
        assert error is not None, f"BBox accepted negative dimensions in {fmt}"
        assert "width and height must be non-negative" in str(error)
