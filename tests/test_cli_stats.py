"""Unit tests for the ``cv-benchkit stats`` subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cv_benchkit.cli import main


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _ann(ann_id: int, image_id: int, category_id: int, w: float, h: float) -> dict[str, Any]:
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [0.0, 0.0, w, h],
        "area": w * h,
    }


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    return _write(
        tmp_path,
        {
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "b.jpg", "width": 640, "height": 480},
                {"id": 3, "file_name": "c.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                _ann(1, 1, 0, 10, 10),  # small, area 100, ratio 1
                _ann(2, 1, 0, 32, 32),  # medium, area exactly 32**2, ratio 1
                _ann(3, 1, 1, 200, 50),  # large, area 10000, ratio 4
                _ann(4, 2, 0, 20, 40),  # small, area 800, ratio 0.5
            ],
            "categories": [
                {"id": 0, "name": "dog"},
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "bird"},
            ],
        },
    )


def _json_stats(path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    assert main(["stats", str(path), "--json"]) == 0
    result: dict[str, Any] = json.loads(capsys.readouterr().out)
    return result


def test_json_overview(sample_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    overview = _json_stats(sample_file, capsys)["overview"]
    assert overview == {
        "images": 3,
        "annotations": 4,
        "categories": 3,
        "mean_annotations_per_image": pytest.approx(4 / 3),
        "median_annotations_per_image": 1.0,
        "unannotated_images": 1,
    }


def test_json_box_scales(sample_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scales = _json_stats(sample_file, capsys)["box_scales"]
    assert scales == {
        "small": {"count": 2, "percent": 50.0},
        "medium": {"count": 1, "percent": 25.0},
        "large": {"count": 1, "percent": 25.0},
    }


def test_json_categories_sorted_by_count(
    sample_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    categories = _json_stats(sample_file, capsys)["categories"]
    assert categories == [
        {"id": 0, "name": "dog", "count": 3, "percent": 75.0},
        {"id": 1, "name": "cat", "count": 1, "percent": 25.0},
        {"id": 2, "name": "bird", "count": 0, "percent": 0.0},
    ]


def test_json_aspect_ratio(sample_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ratio = _json_stats(sample_file, capsys)["aspect_ratio"]
    assert ratio == {
        "min": 0.5,
        "median": 1.0,
        "max": 4.0,
        "mean": pytest.approx((1 + 1 + 4 + 0.5) / 4),
    }


def test_text_output(sample_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["stats", str(sample_file)]) == 0
    out = capsys.readouterr().out
    assert "Images:                    3" in out
    assert "Unannotated images:        1" in out
    assert "mean 1.33, median 1.00" in out
    assert "50.00%" in out
    assert "dog" in out and "75.00%" in out
    assert "min 0.50, median 1.00, max 4.00, mean 1.62" in out


def test_dataset_without_annotations(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(
        tmp_path,
        {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            "annotations": [],
            "categories": [{"id": 0, "name": "dog"}],
        },
    )
    stats = _json_stats(path, capsys)
    assert stats["overview"]["annotations"] == 0
    assert stats["overview"]["unannotated_images"] == 1
    assert stats["overview"]["mean_annotations_per_image"] == 0.0
    assert stats["box_scales"]["small"] == {"count": 0, "percent": 0.0}
    assert stats["categories"] == [{"id": 0, "name": "dog", "count": 0, "percent": 0.0}]
    assert stats["aspect_ratio"] == {"min": None, "median": None, "max": None, "mean": None}

    assert main(["stats", str(path)]) == 0
    assert "min n/a, median n/a, max n/a, mean n/a" in capsys.readouterr().out


def test_empty_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path, {"images": [], "annotations": [], "categories": []})
    assert main(["stats", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Images:                    0" in out
    assert "(no categories)" in out


def test_zero_height_box_skipped_for_aspect_ratio(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(
        tmp_path,
        {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            "annotations": [_ann(1, 1, 0, 5, 0), _ann(2, 1, 0, 6, 3)],
            "categories": [{"id": 0, "name": "dog"}],
        },
    )
    stats = _json_stats(path, capsys)
    assert stats["aspect_ratio"]["min"] == 2.0
    assert stats["aspect_ratio"]["max"] == 2.0
    assert stats["box_scales"]["small"]["count"] == 2


def test_unknown_category_is_listed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(
        tmp_path,
        {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            "annotations": [_ann(1, 1, 7, 2, 2)],
            "categories": [{"id": 0, "name": "dog"}],
        },
    )
    categories = _json_stats(path, capsys)["categories"]
    assert {"id": 7, "name": "<unknown id 7>", "count": 1, "percent": 100.0} in categories


def test_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["stats", str(tmp_path / "missing.json")]) == 1
    assert "not found" in capsys.readouterr().err


def test_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["stats", str(tmp_path)]) == 1
    assert "is a directory" in capsys.readouterr().err


def test_invalid_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert main(["stats", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Validation FAILED" in captured.err


def test_invalid_coco(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path, {"images": [], "annotations": [{"id": 1}], "categories": []})
    assert main(["stats", str(path), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Validation FAILED" in captured.err
