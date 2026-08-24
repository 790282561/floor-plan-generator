#!/usr/bin/env python3
"""Generate a wall-only DXF from a calibrated floor-plan raster image."""

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


def snap_nearby_coordinates(values: list[int], tolerance: int = 3) -> dict[int, int]:
    """Map near-identical raster coordinates to one stable grid coordinate."""
    groups: list[list[int]] = []
    for value in sorted(set(values)):
        if not groups or value - round(sum(groups[-1]) / len(groups[-1])) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    mapping: dict[int, int] = {}
    for group in groups:
        target = int(round(float(np.median(group))))
        mapping.update({value: target for value in group})
    return mapping


def remove_redundant_vertices(
    points: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Remove duplicate and collinear vertices from a closed orthogonal loop."""
    cleaned: list[tuple[int, int]] = []
    for point in points:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()

    changed = True
    while changed and len(cleaned) >= 4:
        changed = False
        result: list[tuple[int, int]] = []
        count = len(cleaned)
        for index, point in enumerate(cleaned):
            previous = cleaned[(index - 1) % count]
            following = cleaned[(index + 1) % count]
            if (
                previous[0] == point[0] == following[0]
                or previous[1] == point[1] == following[1]
            ):
                changed = True
                continue
            result.append(point)
        cleaned = result
    return cleaned


def remove_short_rectangular_tabs(
    points: list[tuple[int, int]], max_depth: int, max_width: int
) -> list[tuple[int, int]]:
    """Flatten short rectangular contour detours caused by wall-line overrun."""
    cleaned = points[:]
    changed = True
    while changed and len(cleaned) >= 6:
        changed = False
        count = len(cleaned)
        for index in range(count):
            b_index = index
            c_index = (index + 1) % count
            d_index = (index + 2) % count
            e_index = (index + 3) % count
            b, c, d, e = (
                cleaned[b_index],
                cleaned[c_index],
                cleaned[d_index],
                cleaned[e_index],
            )
            horizontal_tab = (
                b[1] == e[1]
                and c[1] == d[1]
                and b[0] == c[0]
                and d[0] == e[0]
                and 0 < abs(c[1] - b[1]) <= max_depth
                and 0 < abs(d[0] - c[0]) <= max_width
            )
            vertical_tab = (
                b[0] == e[0]
                and c[0] == d[0]
                and b[1] == c[1]
                and d[1] == e[1]
                and 0 < abs(c[0] - b[0]) <= max_depth
                and 0 < abs(d[1] - c[1]) <= max_width
            )
            if horizontal_tab or vertical_tab:
                for remove_index in sorted({c_index, d_index}, reverse=True):
                    cleaned.pop(remove_index)
                cleaned = remove_redundant_vertices(cleaned)
                changed = True
                break
    return cleaned


def estimate_wall_thickness_px(mask: np.ndarray) -> float:
    """Estimate the common solid-strip thickness from the distance field."""
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    positive = distance[distance > 0]
    if positive.size == 0:
        return 1.0
    return max(1.0, float(np.percentile(positive, 90)) * 2.0)


def extract_closed_wall_polylines(
    roi: np.ndarray,
    x_offset: int,
    y_offset: int,
    max_tab_depth: int,
    max_tab_width: int,
) -> list[list[tuple[int, int]]]:
    """Trace solid wall regions as closed, snapped orthogonal boundaries."""
    # Heal threshold/anti-alias pinholes only. The 3 px kernel is much smaller
    # than a wall opening, so intentional openings remain open.
    clean = cv2.morphologyEx(
        roi, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    contours, _ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    raw_polylines: list[list[tuple[int, int]]] = []
    for contour in contours:
        if abs(cv2.contourArea(contour)) < 20:
            continue
        simplified = cv2.approxPolyDP(contour, 2.0, True).reshape(-1, 2)
        points = [(int(x) + x_offset, int(y) + y_offset) for x, y in simplified]
        if len(points) >= 4:
            raw_polylines.append(points)

    x_map = snap_nearby_coordinates(
        [point[0] for polygon in raw_polylines for point in polygon]
    )
    y_map = snap_nearby_coordinates(
        [point[1] for polygon in raw_polylines for point in polygon]
    )
    polylines: list[list[tuple[int, int]]] = []
    for polygon in raw_polylines:
        snapped = [(x_map[x], y_map[y]) for x, y in polygon]
        cleaned = remove_redundant_vertices(snapped)
        cleaned = remove_short_rectangular_tabs(
            cleaned, max_depth=max_tab_depth, max_width=max_tab_width
        )
        if len(cleaned) >= 4:
            polylines.append(cleaned)
    return polylines


def extract_wall_geometry(
    gray: np.ndarray,
) -> tuple[
    list[tuple[int, int, int, int]], list[list[tuple[int, int]]], dict
]:
    x1, y1, x2, y2 = WALL_BBOX_PX
    roi = np.uint8(gray[y1 : y2 + 1, x1 : x2 + 1] < 190) * 255
    roi = keep_wall_components(roi)

    # CAD exports arrive in two common forms: thin outline linework and solid
    # black wall strips.  Erosion removes thin strokes but preserves the core
    # of solid strips, which provides a stable automatic mode switch.
    ink_pixels = int(np.count_nonzero(roi))
    eroded = cv2.erode(roi, np.ones((5, 5), np.uint8))
    solid_ratio = np.count_nonzero(eroded) / max(ink_pixels, 1)
    if solid_ratio >= 0.12:
        extraction_mode = "solid-wall-boundaries"
        estimated_thickness = estimate_wall_thickness_px(roi)
        max_tab_depth = max(3, int(round(estimated_thickness * 0.75)))
        max_tab_width = max(6, int(round(estimated_thickness * 1.6)))
        polylines = extract_closed_wall_polylines(
            roi, x1, y1, max_tab_depth, max_tab_width
        )
        return [], polylines, {
            "horizontal_boundary_lines": 0,
            "vertical_boundary_lines": 0,
            "closed_wall_polylines": len(polylines),
            "closed_wall_boundary_edges": sum(len(polyline) for polyline in polylines),
            "extraction_mode": extraction_mode,
            "solid_wall_ratio": round(float(solid_ratio), 4),
            "estimated_wall_thickness_px": round(estimated_thickness, 2),
            "tab_cleanup_max_depth_px": max_tab_depth,
            "tab_cleanup_max_width_px": max_tab_width,
            "wall_bbox_px": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }
    else:
        extraction_mode = "thin-outline-lines"
        line_source = roi

    horizontal_mask = cv2.morphologyEx(
        line_source, cv2.MORPH_OPEN, np.ones((1, 7), np.uint8)
    )
    vertical_mask = cv2.morphologyEx(
        line_source, cv2.MORPH_OPEN, np.ones((7, 1), np.uint8)
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
    return lines, [], {
        "horizontal_boundary_lines": len(horizontal),
        "vertical_boundary_lines": len(vertical),
        "extraction_mode": extraction_mode,
        "solid_wall_ratio": round(float(solid_ratio), 4),
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
    path: Path,
    lines: list[tuple[int, int, int, int]],
    polylines: list[list[tuple[int, int]]],
    width: int = 1200,
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
    for polyline in polylines:
        cad_points = [to_cad(point) for point in polyline]
        image_points = [
            (margin + x * scale, height - margin - y * scale)
            for x, y in cad_points
        ]
        draw.line(image_points + [image_points[0]], fill="black", width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def generate(source: Path, output: Path) -> dict:
    gray = read_gray(source)
    lines, polylines, stats = extract_wall_geometry(gray)
    if not lines and not polylines:
        raise RuntimeError("未提取到墙体边界线")

    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    doc.layers.add("WALLS", color=7, linetype="CONTINUOUS")
    doc.layers.add("INNER_WALLS", color=7, linetype="CONTINUOUS")
    doc.layers.add("DIMENSIONS", color=1, linetype="CONTINUOUS")
    msp = doc.modelspace()
    for x1, y1, x2, y2 in lines:
        msp.add_line(to_cad((x1, y1)), to_cad((x2, y2)), dxfattribs={"layer": "WALLS"})
    for polyline in polylines:
        msp.add_lwpolyline(
            [to_cad(point) for point in polyline],
            close=True,
            dxfattribs={"layer": "WALLS"},
        )

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
        "wall_closed_polylines": len(polylines),
        "wall_boundary_edges": len(lines) + sum(len(polyline) for polyline in polylines),
        "wall_topology": "closed-polylines" if polylines else "independent-lines",
        "audit_errors": len(auditor.errors),
        "audit_fixes": len(auditor.fixes),
        **stats,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_preview(output.with_suffix(".png"), lines, polylines)
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
