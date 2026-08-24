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
WALL_WIDTHS_MM = [240.0, 180.0, 120.0]

# Verified against ref_pic_2.jpg: outermost wall-outline extents only.  The
# dimension strings and extension lines lie outside this rectangle.
WALL_BBOX_PX = (436, 536, 1465, 1867)


def load_wall_widths(path: Path) -> list[float]:
    """Load allowed wall widths, accepting the former misspelled key."""
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("wall_width", data.get("wall_wdith"))
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path}: wall_width must be a non-empty list")
    widths = sorted({float(value) for value in values}, reverse=True)
    if any(value <= 0 for value in widths):
        raise ValueError(f"{path}: wall_width values must be positive")
    return widths


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
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Remove duplicate and collinear vertices from a closed orthogonal loop."""
    cleaned: list[tuple[float, float]] = []
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


def dimension_wall_bands() -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Return exact X/Y wall bands explicitly present in dimension chains."""
    allowed = set(WALL_WIDTHS_MM)
    x_bands: list[tuple[float, float, float]] = []
    position = 0.0
    for value in HORIZONTAL_CHAIN_MM:
        end = position + value
        if value in allowed:
            x_bands.append((position, end, value))
        position = end

    y_bands: list[tuple[float, float, float]] = []
    top = 0.0
    for value in VERTICAL_CHAIN_TOP_DOWN_MM:
        bottom = top + value
        if value in allowed:
            y_bands.append(
                (OVERALL_HEIGHT_MM - bottom, OVERALL_HEIGHT_MM - top, value)
            )
        top = bottom
    return x_bands, y_bands


def nearest_width(value: float, choices: list[float]) -> float:
    return min(choices, key=lambda choice: (abs(choice - value), choice))


def snap_rectangle_junctions(
    rectangles: list[tuple[float, float, float, float]],
    assignments: list[dict],
    tolerance: float = 80.0,
) -> list[tuple[float, float, float, float]]:
    """Align wall-strip endpoints with the outer face of intersecting strips."""
    adjusted = [list(rectangle) for rectangle in rectangles]
    horizontal_indices = [
        index
        for index, assignment in enumerate(assignments)
        if assignment["orientation"] == "horizontal"
    ]
    vertical_indices = [
        index
        for index, assignment in enumerate(assignments)
        if assignment["orientation"] == "vertical"
    ]
    for horizontal_index in horizontal_indices:
        hx1, hy1, hx2, hy2 = adjusted[horizontal_index]
        for vertical_index in vertical_indices:
            vx1, vy1, vx2, vy2 = adjusted[vertical_index]
            if hy2 < vy1 - tolerance or hy1 > vy2 + tolerance:
                continue
            if vx1 - tolerance <= hx1 <= vx2 + tolerance and hx2 > vx2:
                hx1 = vx1
            if vx1 - tolerance <= hx2 <= vx2 + tolerance and hx1 < vx1:
                hx2 = vx2
            if hy1 - tolerance <= vy1 <= hy2 + tolerance and vy2 > hy2:
                vy1 = hy1
            if hy1 - tolerance <= vy2 <= hy2 + tolerance and vy1 < hy1:
                vy2 = hy2
            adjusted[vertical_index] = [vx1, vy1, vx2, vy2]
        adjusted[horizontal_index] = [hx1, hy1, hx2, hy2]

    result = [tuple(round(value, 3) for value in rectangle) for rectangle in adjusted]
    for assignment, rectangle in zip(assignments, result):
        assignment["rectangle_mm"] = rectangle
    return result


def matching_dimension_band(
    actual_low: float,
    actual_high: float,
    bands: list[tuple[float, float, float]],
    tolerance: float = 60.0,
) -> tuple[float, float, float] | None:
    actual_center = (actual_low + actual_high) / 2.0
    matches = [
        band
        for band in bands
        if max(actual_low, band[0]) <= min(actual_high, band[1]) + tolerance
    ]
    if not matches:
        return None
    return min(matches, key=lambda band: abs(actual_center - (band[0] + band[1]) / 2.0))


def extract_standard_wall_rectangles(
    roi: np.ndarray, estimated_thickness: float
) -> tuple[list[tuple[float, float, float, float]], list[dict]]:
    """Rebuild axis-aligned wall strips with widths selected from data.json."""
    roi_height, roi_width = roi.shape
    kernel_length = max(7, int(round(estimated_thickness * 2.0)))
    x_bands, y_bands = dimension_wall_bands()
    minimum_width = min(WALL_WIDTHS_MM)
    exterior_widths = [value for value in WALL_WIDTHS_MM if value > minimum_width]
    if not exterior_widths:
        exterior_widths = WALL_WIDTHS_MM[:]
    boundary_tolerance = max(3, int(round(estimated_thickness * 0.25)))
    rectangles: list[tuple[float, float, float, float]] = []
    assignments: list[dict] = []

    orientations = (
        ("horizontal", np.ones((1, kernel_length), np.uint8)),
        ("vertical", np.ones((kernel_length, 1), np.uint8)),
    )
    for orientation, kernel in orientations:
        opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
        for index in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[index])
            if orientation == "horizontal":
                if width < kernel_length or height > estimated_thickness * 2.0:
                    continue
                cad_x1, cad_y_high = to_cad((WALL_BBOX_PX[0] + x, WALL_BBOX_PX[1] + y))
                cad_x2, cad_y_low = to_cad(
                    (WALL_BBOX_PX[0] + x + width - 1, WALL_BBOX_PX[1] + y + height - 1)
                )
                actual_low, actual_high = sorted((cad_y_low, cad_y_high))
                measured_width = actual_high - actual_low
                explicit = matching_dimension_band(actual_low, actual_high, y_bands)
                touches_low = y + height - 1 >= roi_height - 1 - boundary_tolerance
                touches_high = y <= boundary_tolerance
                if explicit is not None:
                    low, high, assigned_width = explicit
                    role, source = "dimensioned", "DIMENSION_CHAIN"
                elif touches_low:
                    assigned_width = nearest_width(measured_width, exterior_widths)
                    low, high = 0.0, assigned_width
                    role, source = "exterior", "IMAGE+EXTERIOR_RULE"
                elif touches_high:
                    assigned_width = nearest_width(measured_width, exterior_widths)
                    low, high = OVERALL_HEIGHT_MM - assigned_width, OVERALL_HEIGHT_MM
                    role, source = "exterior", "IMAGE+EXTERIOR_RULE"
                else:
                    assigned_width = minimum_width
                    center = (actual_low + actual_high) / 2.0
                    low, high = center - assigned_width / 2.0, center + assigned_width / 2.0
                    role, source = "interior", "INTERIOR_RULE"
                rectangle = (min(cad_x1, cad_x2), low, max(cad_x1, cad_x2), high)
            else:
                if height < kernel_length or width > estimated_thickness * 2.0:
                    continue
                cad_x_low, cad_y1 = to_cad((WALL_BBOX_PX[0] + x, WALL_BBOX_PX[1] + y))
                cad_x_high, cad_y2 = to_cad(
                    (WALL_BBOX_PX[0] + x + width - 1, WALL_BBOX_PX[1] + y + height - 1)
                )
                actual_low, actual_high = sorted((cad_x_low, cad_x_high))
                measured_width = actual_high - actual_low
                explicit = matching_dimension_band(actual_low, actual_high, x_bands)
                touches_low = x <= boundary_tolerance
                touches_high = x + width - 1 >= roi_width - 1 - boundary_tolerance
                if explicit is not None:
                    low, high, assigned_width = explicit
                    role, source = "dimensioned", "DIMENSION_CHAIN"
                elif touches_low:
                    assigned_width = nearest_width(measured_width, exterior_widths)
                    low, high = 0.0, assigned_width
                    role, source = "exterior", "IMAGE+EXTERIOR_RULE"
                elif touches_high:
                    assigned_width = nearest_width(measured_width, exterior_widths)
                    low, high = OVERALL_WIDTH_MM - assigned_width, OVERALL_WIDTH_MM
                    role, source = "exterior", "IMAGE+EXTERIOR_RULE"
                else:
                    assigned_width = minimum_width
                    center = (actual_low + actual_high) / 2.0
                    low, high = center - assigned_width / 2.0, center + assigned_width / 2.0
                    role, source = "interior", "INTERIOR_RULE"
                rectangle = (low, min(cad_y1, cad_y2), high, max(cad_y1, cad_y2))

            rectangle = tuple(round(value, 3) for value in rectangle)
            rectangles.append(rectangle)
            assignments.append(
                {
                    "orientation": orientation,
                    "pixel_bbox": {"x": x, "y": y, "width": width, "height": height},
                    "measured_width_mm": round(measured_width, 3),
                    "assigned_width_mm": assigned_width,
                    "role": role,
                    "source": source,
                    "rectangle_mm": rectangle,
                    "pixel_area": area,
                }
            )
    rectangles = snap_rectangle_junctions(rectangles, assignments)
    return rectangles, assignments


def rectangles_to_closed_polylines(
    rectangles: list[tuple[float, float, float, float]],
) -> list[list[tuple[float, float]]]:
    """Compute exact orthogonal union boundaries without an extra dependency."""
    xs = sorted({value for rectangle in rectangles for value in (rectangle[0], rectangle[2])})
    ys = sorted({value for rectangle in rectangles for value in (rectangle[1], rectangle[3])})
    occupied = [[False for _ in range(len(ys) - 1)] for _ in range(len(xs) - 1)]
    for i in range(len(xs) - 1):
        center_x = (xs[i] + xs[i + 1]) / 2.0
        for j in range(len(ys) - 1):
            center_y = (ys[j] + ys[j + 1]) / 2.0
            occupied[i][j] = any(
                x1 < center_x < x2 and y1 < center_y < y2
                for x1, y1, x2, y2 in rectangles
            )

    edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            if not occupied[i][j]:
                continue
            x1, x2, y1, y2 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            if j == 0 or not occupied[i][j - 1]:
                edges.add(((x1, y1), (x2, y1)))
            if i == len(xs) - 2 or not occupied[i + 1][j]:
                edges.add(((x2, y1), (x2, y2)))
            if j == len(ys) - 2 or not occupied[i][j + 1]:
                edges.add(((x2, y2), (x1, y2)))
            if i == 0 or not occupied[i - 1][j]:
                edges.add(((x1, y2), (x1, y1)))

    outgoing: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for start, end in edges:
        outgoing.setdefault(start, []).append(end)
    remaining = set(edges)
    loops: list[list[tuple[float, float]]] = []
    while remaining:
        first_start, first_end = next(iter(remaining))
        loop = [first_start]
        previous, current = first_start, first_end
        remaining.remove((first_start, first_end))
        loop.append(current)
        while current != first_start:
            candidates = [end for end in outgoing.get(current, []) if (current, end) in remaining]
            if not candidates:
                raise RuntimeError(f"Open wall union boundary at {current}")
            if len(candidates) == 1:
                following = candidates[0]
            else:
                incoming = (current[0] - previous[0], current[1] - previous[1])
                following = max(
                    candidates,
                    key=lambda point: (
                        incoming[0] * (point[1] - current[1])
                        - incoming[1] * (point[0] - current[0]),
                        incoming[0] * (point[0] - current[0])
                        + incoming[1] * (point[1] - current[1]),
                    ),
                )
            remaining.remove((current, following))
            previous, current = current, following
            if current != first_start:
                loop.append(current)
        simplified = remove_redundant_vertices(loop)
        if len(simplified) >= 4:
            loops.append(simplified)
    return loops


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
    list[tuple[int, int, int, int]], list[list[tuple[float, float]]], dict
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
        rectangles, assignments = extract_standard_wall_rectangles(
            roi, estimated_thickness
        )
        polylines = rectangles_to_closed_polylines(rectangles)
        return [], polylines, {
            "horizontal_boundary_lines": 0,
            "vertical_boundary_lines": 0,
            "closed_wall_polylines": len(polylines),
            "closed_wall_boundary_edges": sum(len(polyline) for polyline in polylines),
            "extraction_mode": extraction_mode,
            "solid_wall_ratio": round(float(solid_ratio), 4),
            "estimated_wall_thickness_px": round(estimated_thickness, 2),
            "wall_width_candidates_mm": WALL_WIDTHS_MM,
            "wall_width_assignments": assignments,
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

    # Thin-outline drawings used to be emitted as independent LINE entities.
    # That breaks the wall-topology contract and leaves later opening stages to
    # guess which edges belong together. Trace the detected wall ink into
    # closed orthogonal boundaries here instead. If no closed boundary can be
    # reconstructed, stop rather than writing a structurally invalid wall DXF.
    traced = extract_closed_wall_polylines(
        roi,
        x_offset=x1,
        y_offset=y1,
        max_tab_depth=max(3, int(round(min(roi.shape) * 0.006))),
        max_tab_width=max(12, int(round(min(roi.shape) * 0.03))),
    )
    polylines = [
        [to_cad(point) for point in polygon]
        for polygon in traced
        if len(polygon) >= 4
    ]
    if not polylines:
        raise RuntimeError("细线墙体无法重建为闭合 LWPOLYLINE，已停止输出")
    return [], polylines, {
        "horizontal_boundary_lines": len(horizontal),
        "vertical_boundary_lines": len(vertical),
        "closed_wall_polylines": len(polylines),
        "closed_wall_boundary_edges": sum(len(polyline) for polyline in polylines),
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
    polylines: list[list[tuple[float, float]]],
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
        cad_points = polyline
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
    if lines:
        raise RuntimeError("墙体重建后仍存在独立 LINE，已停止输出")
    if not polylines:
        raise RuntimeError("未提取到闭合墙体多段线")

    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    doc.layers.add("WALLS", color=7, linetype="CONTINUOUS")
    doc.layers.add("INNER_WALLS", color=7, linetype="CONTINUOUS")
    doc.layers.add("DIMENSIONS", color=1, linetype="CONTINUOUS")
    msp = doc.modelspace()
    for polyline in polylines:
        msp.add_lwpolyline(
            polyline,
            close=True,
            dxfattribs={"layer": "WALLS"},
        )

    add_dimensions(msp, add_dimension_style(doc))
    auditor = doc.audit()
    wall_lines = sum(
        1
        for entity in msp
        if entity.dxf.layer in {"WALLS", "INNER_WALLS"}
        and entity.dxftype() == "LINE"
    )
    open_wall_polylines = sum(
        1
        for entity in msp
        if entity.dxf.layer in {"WALLS", "INNER_WALLS"}
        and entity.dxftype() == "LWPOLYLINE"
        and not entity.closed
    )
    closed_wall_polylines = sum(
        1
        for entity in msp
        if entity.dxf.layer in {"WALLS", "INNER_WALLS"}
        and entity.dxftype() == "LWPOLYLINE"
        and entity.closed
    )
    wall_geometry_valid = (
        wall_lines == 0
        and open_wall_polylines == 0
        and closed_wall_polylines > 0
        and len(auditor.errors) == 0
    )
    if not wall_geometry_valid:
        raise RuntimeError(
            "墙体闭合校核失败："
            f"LINE={wall_lines}, OPEN_LWPOLYLINE={open_wall_polylines}, "
            f"CLOSED_LWPOLYLINE={closed_wall_polylines}, AUDIT_ERRORS={len(auditor.errors)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(output)
    report = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "overall_width_mm": OVERALL_WIDTH_MM,
        "overall_height_mm": OVERALL_HEIGHT_MM,
        "horizontal_chain_mm": HORIZONTAL_CHAIN_MM,
        "vertical_chain_top_down_mm": VERTICAL_CHAIN_TOP_DOWN_MM,
        "wall_boundary_lines": wall_lines,
        "wall_open_polylines": open_wall_polylines,
        "wall_closed_polylines": closed_wall_polylines,
        "wall_boundary_edges": sum(len(polyline) for polyline in polylines),
        "wall_topology": "closed-lwpolylines",
        "wall_geometry_valid": wall_geometry_valid,
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
    global VERTICAL_CHAIN_TOP_DOWN_MM, WALL_BBOX_PX, WALL_WIDTHS_MM
    parser = argparse.ArgumentParser(description="从 ref_pic_2 生成仅含墙体和尺寸的 DXF")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("data.json"),
        help="JSON config containing wall_width candidates in millimetres",
    )
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
    WALL_WIDTHS_MM = load_wall_widths(args.data)
    if abs(sum(HORIZONTAL_CHAIN_MM) - OVERALL_WIDTH_MM) > 0.01:
        parser.error("横向尺寸链之和必须等于 overall-width-mm")
    if abs(sum(VERTICAL_CHAIN_TOP_DOWN_MM) - OVERALL_HEIGHT_MM) > 0.01:
        parser.error("纵向尺寸链之和必须等于 overall-height-mm")
    print(json.dumps(generate(args.source, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
