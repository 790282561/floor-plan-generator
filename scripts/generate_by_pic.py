#!/usr/bin/env python3
"""Generate a wall-only DXF from the checked ref_pic_2 CAD raster export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import ezdxf
import numpy as np
from ezdxf.enums import TextEntityAlignment
from PIL import Image, ImageDraw


OVERALL_WIDTH_MM = 10700.0
OVERALL_HEIGHT_MM = 13990.0
HORIZONTAL_CHAIN_MM = [2240.0, 3360.0, 2700.0, 1200.0, 1200.0]
VERTICAL_CHAIN_TOP_DOWN_MM = [740.0, 3500.0, 5000.0, 3500.0, 1250.0]

# Verified against ref_pic_2.jpg: outermost wall-outline extents only.  The
# dimension strings and extension lines lie outside this rectangle.
WALL_BBOX_PX = (436, 536, 1465, 1867)


def read_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def keep_wall_components(mask: np.ndarray) -> np.ndarray:
    """Drop isolated reference markers while preserving every wall piece."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    kept = np.zeros_like(mask)
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area < 8:
            continue
        if width < 24 and height < 24:
            continue
        kept[labels == index] = 255
    return kept


def merge_axis_segments(
    items: list[tuple[int, int, int]], axis_tolerance: int = 2, gap: int = 3
) -> list[tuple[int, int, int]]:
    """Merge overlapping collinear raster runs without bridging wall openings."""
    merged: list[tuple[int, int, int]] = []
    for axis, start, end in sorted(items):
        match = None
        for index, (old_axis, old_start, old_end) in enumerate(merged):
            if (
                abs(old_axis - axis) <= axis_tolerance
                and start <= old_end + gap
                and end >= old_start - gap
            ):
                match = index
                break
        if match is None:
            merged.append((axis, start, end))
        else:
            old_axis, old_start, old_end = merged[match]
            merged[match] = (
                int(round((old_axis + axis) / 2)),
                min(old_start, start),
                max(old_end, end),
            )
    return sorted(merged)


def extract_axis_lines(gray: np.ndarray) -> tuple[list[tuple[int, int, int, int]], dict]:
    x1, y1, x2, y2 = WALL_BBOX_PX
    roi = np.uint8(gray[y1 : y2 + 1, x1 : x2 + 1] < 190) * 255
    roi = keep_wall_components(roi)

    horizontal_mask = cv2.morphologyEx(
        roi, cv2.MORPH_OPEN, np.ones((1, 7), np.uint8)
    )
    vertical_mask = cv2.morphologyEx(
        roi, cv2.MORPH_OPEN, np.ones((7, 1), np.uint8)
    )

    horizontal: list[tuple[int, int, int]] = []
    count, _, stats, _ = cv2.connectedComponentsWithStats(horizontal_mask, 8)
    for index in range(1, count):
        x, y, width, height, _ = (int(value) for value in stats[index])
        if width >= 7 and height <= 6:
            horizontal.append((y1 + y + height // 2, x1 + x, x1 + x + width - 1))

    vertical: list[tuple[int, int, int]] = []
    count, _, stats, _ = cv2.connectedComponentsWithStats(vertical_mask, 8)
    for index in range(1, count):
        x, y, width, height, _ = (int(value) for value in stats[index])
        if height >= 7 and width <= 6:
            vertical.append((x1 + x + width // 2, y1 + y, y1 + y + height - 1))

    horizontal = merge_axis_segments(horizontal)
    vertical = merge_axis_segments(vertical)
    lines = [(start, axis, end, axis) for axis, start, end in horizontal]
    lines.extend((axis, start, axis, end) for axis, start, end in vertical)
    return lines, {
        "horizontal_boundary_lines": len(horizontal),
        "vertical_boundary_lines": len(vertical),
        "wall_bbox_px": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


def to_cad(point: tuple[int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = WALL_BBOX_PX
    px, py = point
    x = (px - x1) * OVERALL_WIDTH_MM / (x2 - x1)
    y = (y2 - py) * OVERALL_HEIGHT_MM / (y2 - y1)
    return round(x, 3), round(y, 3)


def cumulative(values: list[float]) -> list[float]:
    result = [0.0]
    for value in values:
        result.append(result[-1] + value)
    return result


def add_dimension_style(doc: ezdxf.document.Drawing) -> str:
    name = "FP-DIM"
    if name not in doc.dimstyles:
        doc.dimstyles.new(
            name,
            dxfattribs={
                "dimtxt": 250,
                "dimasz": 160,
                "dimexo": 100,
                "dimexe": 120,
                "dimgap": 80,
                "dimdec": 0,
                "dimclrd": 1,
                "dimclre": 1,
                "dimclrt": 1,
            },
        )
    return name


def add_dimensions(msp, dimstyle: str) -> None:
    x_chain = cumulative(HORIZONTAL_CHAIN_MM)
    y_chain = cumulative(list(reversed(VERTICAL_CHAIN_TOP_DOWN_MM)))

    for start, end in zip(x_chain, x_chain[1:]):
        msp.add_linear_dim(
            base=(0, OVERALL_HEIGHT_MM + 650),
            p1=(start, OVERALL_HEIGHT_MM),
            p2=(end, OVERALL_HEIGHT_MM),
            angle=0,
            dimstyle=dimstyle,
            dxfattribs={"layer": "DIMENSIONS"},
        ).render()
    msp.add_linear_dim(
        base=(0, OVERALL_HEIGHT_MM + 1350),
        p1=(0, OVERALL_HEIGHT_MM),
        p2=(OVERALL_WIDTH_MM, OVERALL_HEIGHT_MM),
        angle=0,
        dimstyle=dimstyle,
        dxfattribs={"layer": "DIMENSIONS"},
    ).render()

    for start, end in zip(y_chain, y_chain[1:]):
        msp.add_linear_dim(
            base=(-650, 0),
            p1=(0, start),
            p2=(0, end),
            angle=90,
            dimstyle=dimstyle,
            dxfattribs={"layer": "DIMENSIONS"},
        ).render()
    msp.add_linear_dim(
        base=(-1350, 0),
        p1=(0, 0),
        p2=(0, OVERALL_HEIGHT_MM),
        angle=90,
        dimstyle=dimstyle,
        dxfattribs={"layer": "DIMENSIONS"},
    ).render()


def write_preview(
    path: Path, lines: list[tuple[int, int, int, int]], width: int = 1200
) -> None:
    margin = 100
    scale = (width - margin * 2) / OVERALL_WIDTH_MM
    height = int(OVERALL_HEIGHT_MM * scale + margin * 2)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2 in lines:
        p1 = to_cad((x1, y1))
        p2 = to_cad((x2, y2))
        a = (margin + p1[0] * scale, height - margin - p1[1] * scale)
        b = (margin + p2[0] * scale, height - margin - p2[1] * scale)
        draw.line((a, b), fill="black", width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def generate(source: Path, output: Path) -> dict:
    gray = read_gray(source)
    lines, stats = extract_axis_lines(gray)
    if not lines:
        raise RuntimeError("未提取到墙体边界线")

    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    doc.layers.add("WALLS", color=7, linetype="CONTINUOUS")
    doc.layers.add("INNER_WALLS", color=7, linetype="CONTINUOUS")
    doc.layers.add("DIMENSIONS", color=1, linetype="CONTINUOUS")
    msp = doc.modelspace()
    for x1, y1, x2, y2 in lines:
        msp.add_line(to_cad((x1, y1)), to_cad((x2, y2)), dxfattribs={"layer": "WALLS"})

    add_dimensions(msp, add_dimension_style(doc))
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(output)

    auditor = doc.audit()
    report = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "overall_width_mm": OVERALL_WIDTH_MM,
        "overall_height_mm": OVERALL_HEIGHT_MM,
        "horizontal_chain_mm": HORIZONTAL_CHAIN_MM,
        "vertical_chain_top_down_mm": VERTICAL_CHAIN_TOP_DOWN_MM,
        "wall_boundary_lines": len(lines),
        "audit_errors": len(auditor.errors),
        "audit_fixes": len(auditor.fixes),
        **stats,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_preview(output.with_suffix(".png"), lines)
    return report


def main() -> int:
    global OVERALL_WIDTH_MM, OVERALL_HEIGHT_MM, HORIZONTAL_CHAIN_MM
    global VERTICAL_CHAIN_TOP_DOWN_MM, WALL_BBOX_PX
    parser = argparse.ArgumentParser(description="从 ref_pic_2 生成仅含墙体和尺寸的 DXF")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overall-width-mm", type=float, default=OVERALL_WIDTH_MM)
    parser.add_argument("--overall-height-mm", type=float, default=OVERALL_HEIGHT_MM)
    parser.add_argument(
        "--wall-bbox",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=WALL_BBOX_PX,
        help="图片中墙体外框像素范围；换项目图片时必须重新测量",
    )
    parser.add_argument(
        "--horizontal-chain-mm",
        nargs="+",
        type=float,
        default=HORIZONTAL_CHAIN_MM,
        help="横向分段尺寸链，和为总宽",
    )
    parser.add_argument(
        "--vertical-chain-top-down-mm",
        nargs="+",
        type=float,
        default=VERTICAL_CHAIN_TOP_DOWN_MM,
        help="纵向分段尺寸链（从上到下），和为总高",
    )
    args = parser.parse_args()
    # Keep the existing function API small while allowing each project to
    # provide its own image calibration from the command line.
    OVERALL_WIDTH_MM = args.overall_width_mm
    OVERALL_HEIGHT_MM = args.overall_height_mm
    HORIZONTAL_CHAIN_MM = list(args.horizontal_chain_mm)
    VERTICAL_CHAIN_TOP_DOWN_MM = list(args.vertical_chain_top_down_mm)
    WALL_BBOX_PX = tuple(args.wall_bbox)
    if abs(sum(HORIZONTAL_CHAIN_MM) - OVERALL_WIDTH_MM) > 0.01:
        parser.error("横向尺寸链之和必须等于 overall-width-mm")
    if abs(sum(VERTICAL_CHAIN_TOP_DOWN_MM) - OVERALL_HEIGHT_MM) > 0.01:
        parser.error("纵向尺寸链之和必须等于 overall-height-mm")
    print(json.dumps(generate(args.source, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
