"""Unit tests for cv_benchkit.cli."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cv_benchkit import __version__
from cv_benchkit.cli import main


class TestCLI:
    """Tests for CLI arguments and subcommands."""

    @pytest.fixture
    def sample_coco_file(self, tmp_path: Path) -> Path:
        coco_data = {
            "images": [
                {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [10.0, 10.0, 20.0, 20.0],
                    "area": 400.0,
                    "iscrowd": 0,
                },
            ],
            "categories": [
                {"id": 0, "name": "dog"},
            ],
        }
        file_path = tmp_path / "coco.json"
        file_path.write_text(json.dumps(coco_data), encoding="utf-8")
        return file_path

    def test_validate_valid_file(self, sample_coco_file: Path) -> None:
        exit_code = main(["validate", str(sample_coco_file)])
        assert exit_code == 0

    def test_validate_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.json"
        exit_code = main(["validate", str(missing)])
        assert exit_code == 1

    def test_validate_directory_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dir_path = tmp_path / "data"
        dir_path.mkdir()
        exit_code = main(["validate", str(dir_path)])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "is a directory, expected a JSON file" in captured.err

    def test_validate_invalid_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json", encoding="utf-8")
        exit_code = main(["validate", str(bad_file)])
        assert exit_code == 1

    def test_convert_coco_to_yolo(self, sample_coco_file: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "yolo_out"
        exit_code = main(
            [
                "convert",
                "--from",
                "coco",
                "--to",
                "yolo",
                "--input",
                str(sample_coco_file),
                "--output",
                str(out_dir),
            ]
        )
        assert exit_code == 0
        assert (out_dir / "test.txt").exists()

    def test_convert_coco_to_voc(self, sample_coco_file: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "voc_out"
        exit_code = main(
            [
                "convert",
                "--from",
                "coco",
                "--to",
                "voc",
                "--input",
                str(sample_coco_file),
                "--output",
                str(out_dir),
            ]
        )
        assert exit_code == 0
        assert (out_dir / "test.xml").exists()

    def test_convert_yolo_to_coco_and_voc(self, sample_coco_file: Path, tmp_path: Path) -> None:
        yolo_dir = tmp_path / "yolo_labels"
        main(
            [
                "convert",
                "--from",
                "coco",
                "--to",
                "yolo",
                "--input",
                str(sample_coco_file),
                "--output",
                str(yolo_dir),
            ]
        )

        # Missing categories flag
        assert (
            main(
                [
                    "convert",
                    "--from",
                    "yolo",
                    "--to",
                    "coco",
                    "--input",
                    str(yolo_dir),
                    "--output",
                    str(tmp_path / "c1"),
                ]
            )
            == 1
        )

        # Missing dimensions
        assert (
            main(
                [
                    "convert",
                    "--from",
                    "yolo",
                    "--to",
                    "coco",
                    "--input",
                    str(yolo_dir),
                    "--output",
                    str(tmp_path / "c2"),
                    "--categories",
                    "dog",
                ]
            )
            == 1
        )

        # Successful yolo to coco
        coco_out = tmp_path / "coco_out"
        assert (
            main(
                [
                    "convert",
                    "--from",
                    "yolo",
                    "--to",
                    "coco",
                    "--input",
                    str(yolo_dir),
                    "--output",
                    str(coco_out),
                    "--image-width",
                    "100",
                    "--image-height",
                    "100",
                    "--categories",
                    "dog",
                ]
            )
            == 0
        )
        assert (coco_out / "annotations.json").exists()

        # Successful yolo to voc
        voc_out = tmp_path / "voc_out_from_yolo"
        assert (
            main(
                [
                    "convert",
                    "--from",
                    "yolo",
                    "--to",
                    "voc",
                    "--input",
                    str(yolo_dir),
                    "--output",
                    str(voc_out),
                    "--image-width",
                    "100",
                    "--image-height",
                    "100",
                    "--categories",
                    "dog",
                ]
            )
            == 0
        )
        assert (voc_out / "test.xml").exists()

    def test_convert_voc_to_coco_and_yolo(self, sample_coco_file: Path, tmp_path: Path) -> None:
        voc_dir = tmp_path / "voc_annotations"
        main(
            [
                "convert",
                "--from",
                "coco",
                "--to",
                "voc",
                "--input",
                str(sample_coco_file),
                "--output",
                str(voc_dir),
            ]
        )

        # Missing categories
        assert (
            main(
                [
                    "convert",
                    "--from",
                    "voc",
                    "--to",
                    "coco",
                    "--input",
                    str(voc_dir),
                    "--output",
                    str(tmp_path / "v1"),
                ]
            )
            == 1
        )

        # Successful voc to coco
        coco_out = tmp_path / "coco_from_voc"
        assert (
            main(
                [
                    "convert",
                    "--from",
                    "voc",
                    "--to",
                    "coco",
                    "--input",
                    str(voc_dir),
                    "--output",
                    str(coco_out),
                    "--categories",
                    "dog",
                ]
            )
            == 0
        )
        assert (coco_out / "annotations.json").exists()

        # Successful voc to yolo
        yolo_out = tmp_path / "yolo_from_voc"
        assert (
            main(
                [
                    "convert",
                    "--from",
                    "voc",
                    "--to",
                    "yolo",
                    "--input",
                    str(voc_dir),
                    "--output",
                    str(yolo_out),
                    "--categories",
                    "dog",
                ]
            )
            == 0
        )
        assert (yolo_out / "test.txt").exists()

    def test_convert_same_format_error(self, sample_coco_file: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        exit_code = main(
            [
                "convert",
                "--from",
                "coco",
                "--to",
                "coco",
                "--input",
                str(sample_coco_file),
                "--output",
                str(out_dir),
            ]
        )
        assert exit_code == 1

    def test_cli_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out

    @pytest.mark.parametrize(
        ("width", "height"),
        [
            ("0", "100"),
            ("100", "0"),
            ("-10", "100"),
            ("100", "-10"),
            ("0", "0"),
            ("-5", "-5"),
        ],
    )
    def test_convert_yolo_invalid_dimensions(
        self,
        sample_coco_file: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        width: str,
        height: str,
    ) -> None:
        yolo_dir = tmp_path / "yolo_in"
        main(
            [
                "convert",
                "--from",
                "coco",
                "--to",
                "yolo",
                "--input",
                str(sample_coco_file),
                "--output",
                str(yolo_dir),
            ]
        )
        exit_code = main(
            [
                "convert",
                "--from",
                "yolo",
                "--to",
                "coco",
                "--input",
                str(yolo_dir),
                "--output",
                str(tmp_path / f"out_invalid_{width}_{height}"),
                "--categories",
                "dog",
                "--image-width",
                width,
                "--image-height",
                height,
            ]
        )
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error: --image-width and --image-height must be positive integers." in captured.err
