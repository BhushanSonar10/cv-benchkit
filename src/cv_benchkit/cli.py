"""CLI entry point for cv-benchkit.

Subcommands
-----------
- ``cv-benchkit convert`` — convert between dataset formats
- ``cv-benchkit validate`` — validate a COCO JSON file
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from cv_benchkit import __version__
from cv_benchkit.converters import (
    coco_to_voc,
    coco_to_yolo,
    validate_coco_file,
    voc_to_coco,
    yolo_to_coco,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cv-benchkit",
        description="Computer-vision benchmark toolkit: convert and validate datasets.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── convert ──────────────────────────────────────────────────────────
    convert_p = sub.add_parser("convert", help="Convert between dataset formats.")
    convert_p.add_argument(
        "--from",
        dest="src_format",
        required=True,
        choices=["coco", "yolo", "voc"],
        help="Source format.",
    )
    convert_p.add_argument(
        "--to",
        dest="dst_format",
        required=True,
        choices=["coco", "yolo", "voc"],
        help="Target format.",
    )
    convert_p.add_argument(
        "--input",
        dest="input_path",
        required=True,
        help="Input file or directory.",
    )
    convert_p.add_argument(
        "--output",
        dest="output_path",
        required=True,
        help="Output directory.",
    )
    convert_p.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Image width (required for YOLO source).",
    )
    convert_p.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Image height (required for YOLO source).",
    )
    convert_p.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Category names (required for YOLO/VOC source).",
    )

    # ── validate ─────────────────────────────────────────────────────────
    validate_p = sub.add_parser("validate", help="Validate a COCO JSON file.")
    validate_p.add_argument("file", help="Path to the COCO JSON file.")

    return parser


def _cmd_convert(args: argparse.Namespace) -> int:
    src = args.src_format
    dst = args.dst_format
    inp = Path(args.input_path)
    out = Path(args.output_path)

    if src == dst:
        print(f"Error: source and target format are the same ({src}).", file=sys.stderr)
        return 1

    if src == "coco":
        coco = validate_coco_file(inp)
        paths = coco_to_yolo(coco, out) if dst == "yolo" else coco_to_voc(coco, out)
        print(f"Wrote {len(paths)} files to {out}")
        return 0

    if src == "yolo":
        cats = args.categories
        if not cats:
            print("Error: --categories required for YOLO source.", file=sys.stderr)
            return 1
        w = args.image_width
        h = args.image_height
        if not w or not h:
            print(
                "Error: --image-width and --image-height required for YOLO source.", file=sys.stderr
            )
            return 1
        stems = [p.stem for p in inp.glob("*.txt")]
        image_sizes = {s: (w, h) for s in stems}
        coco = yolo_to_coco(inp, image_sizes, cats)
        if dst == "coco":
            out.mkdir(parents=True, exist_ok=True)
            json_path = out / "annotations.json"
            json_path.write_text(
                json.dumps(coco.model_dump(), indent=2),
                encoding="utf-8",
            )
            print(f"Wrote {json_path}")
        else:  # voc
            paths = coco_to_voc(coco, out)
            print(f"Wrote {len(paths)} files to {out}")
        return 0

    if src == "voc":
        cats = args.categories
        if not cats:
            print("Error: --categories required for VOC source.", file=sys.stderr)
            return 1
        coco = voc_to_coco(inp, cats)
        if dst == "coco":
            out.mkdir(parents=True, exist_ok=True)
            json_path = out / "annotations.json"
            json_path.write_text(
                json.dumps(coco.model_dump(), indent=2),
                encoding="utf-8",
            )
            print(f"Wrote {json_path}")
        else:  # yolo
            paths = coco_to_yolo(coco, out)
            print(f"Wrote {len(paths)} files to {out}")
        return 0

    return 1  # pragma: no cover


def _cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"Error: file {path} not found.", file=sys.stderr)
        return 1
    try:
        coco = validate_coco_file(path)
    except Exception as exc:  # noqa: BLE001
        print(f"Validation FAILED: {exc}", file=sys.stderr)
        return 1
    msg = (
        f"Valid COCO dataset: {len(coco.images)} images, "
        f"{len(coco.annotations)} annotations, {len(coco.categories)} categories."
    )
    print(msg)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "convert":
        return _cmd_convert(args)
    if args.command == "validate":
        return _cmd_validate(args)

    parser.print_help()
    return 1  # pragma: no cover
